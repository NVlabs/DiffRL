# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os

from torch.nn.utils.clip_grad import clip_grad_norm_

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import time
import copy
from tensorboardX import SummaryWriter
import yaml

from shac import envs
import shac.models.actor as actor_models
import shac.models.critic as critic_models
from shac.utils.common import *
import shac.utils.torch_utils as tu
from shac.utils.running_mean_std import RunningMeanStd
from shac.utils.dataset import CriticDataset, QCriticDataset
from shac.utils.time_report import TimeReport
from shac.utils.average_meter import AverageMeter


class AHAC:
    def __init__(self, cfg):
        env_name = cfg["params"]["diff_env"].pop("name")
        env_fn = getattr(envs, env_name)

        if "stochastic_init" in cfg["params"]["diff_env"]:
            stochastic_init = cfg["params"]["diff_env"].pop("stochastic_init")
        else:
            stochastic_init = True

        config = dict(
            num_envs=cfg["params"]["config"]["num_actors"],
            device=cfg["params"]["general"]["device"],
            render=cfg["params"]["general"]["render"],
            seed=cfg["params"]["general"]["seed"],
            episode_length=cfg["params"]["diff_env"].get("episode_length", 250),
            stochastic_init=stochastic_init,
            no_grad=False,
        )

        config.update(cfg["params"].get("diff_env", {}))
        seeding(config["seed"])

        self.env = env_fn(**config)
        # reset diff_env config for yaml
        cfg["params"]["diff_env"] = config
        cfg["params"]["diff_env"]["name"] = env_name

        print("num_envs = ", self.env.num_envs)
        print("num_actions = ", self.env.num_actions)
        print("num_obs = ", self.env.num_obs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length
        self.device = cfg["params"]["general"]["device"]

        self.gamma = cfg["params"]["config"].get("gamma", 0.99)

        self.critic_method = cfg["params"]["config"]["critic_method"]
        assert self.critic_method in ["one-step", "td-lambda"]
        self.lam = cfg["params"]["config"].get("lambda", 0.95)

        self.steps_min = cfg["params"]["config"].get("steps_min", 0)
        self.steps_num = cfg["params"]["config"]["steps_num"]
        self.contact_th = cfg["params"]["config"].get("contact_theshold", 1e9)
        self.max_epochs = cfg["params"]["config"]["max_epochs"]
        self.actor_lr = float(cfg["params"]["config"]["actor_learning_rate"])
        self.critic_lr = float(cfg["params"]["config"]["critic_learning_rate"])
        self.lr_schedule = cfg["params"]["config"].get("lr_schedule", "linear")

        self.target_critic_alpha = cfg["params"]["config"].get(
            "target_critic_alpha", 0.4
        )

        self.obs_rms = None
        if cfg["params"]["config"].get("obs_rms", False):
            self.obs_rms = RunningMeanStd(shape=(self.num_obs), device=self.device)

        self.ret_rms = None
        if cfg["params"]["config"].get("ret_rms", False):
            self.ret_rms = RunningMeanStd(shape=(), device=self.device)

        self.critic_iterations = cfg["params"]["config"].get("critic_iterations", 16)
        self.num_batch = cfg["params"]["config"].get("num_batch", 4)
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.name = cfg["params"]["config"].get("name", "Ant")

        self.truncate_grad = cfg["params"]["config"]["truncate_grads"]
        self.grad_norm = cfg["params"]["config"]["grad_norm"]

        if cfg["params"]["general"]["train"]:
            self.log_dir = cfg["params"]["general"]["logdir"]
            os.makedirs(self.log_dir, exist_ok=True)
            # save config
            save_cfg = copy.deepcopy(cfg)
            if "general" in save_cfg["params"]:
                deleted_keys = []
                for key in save_cfg["params"]["general"].keys():
                    if key in save_cfg["params"]["config"]:
                        deleted_keys.append(key)
                for key in deleted_keys:
                    del save_cfg["params"]["general"][key]

            yaml.dump(save_cfg, open(os.path.join(self.log_dir, "cfg.yaml"), "w"))
            self.writer = SummaryWriter(os.path.join(self.log_dir, "log"))
            # save interval
            self.save_interval = cfg["params"]["config"].get("save_interval", 500)
            # stochastic inference
            self.stochastic_evaluation = True
        else:
            self.stochastic_evaluation = not (
                cfg["params"]["config"]["player"].get("determenistic", False)
                or cfg["params"]["config"]["player"].get("deterministic", False)
            )
            self.steps_num = self.env.episode_length

        self.eval_runs = cfg["params"]["config"]["player"]["games_num"]

        # create actor critic network
        # choices: ['ActorDeterministicMLP', 'ActorStochasticMLP']
        self.actor_name = cfg["params"]["network"].get("actor", "ActorStochasticMLP")
        actor_fn = getattr(actor_models, self.actor_name)
        self.actor = actor_fn(
            self.num_obs, self.num_actions, cfg["params"]["network"], device=self.device
        )
        self.critic_name = "CriticMLP"  # NOTE: hardcoded for future proofness
        critic_fn = getattr(critic_models, self.critic_name)
        self.critic = critic_fn(
            self.num_obs, cfg["params"]["network"], device=self.device
        )
        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.target_critic = copy.deepcopy(self.critic)

        if cfg["params"]["general"]["train"]:
            self.save("init_policy")

        # initialize optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            betas=cfg["params"]["config"]["betas"],
            lr=self.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            betas=cfg["params"]["config"]["betas"],
            lr=self.critic_lr,
        )

        # accumulate rewards for each environment
        # TODO make this vectorized somehow
        self.rew_acc = [
            torch.tensor([0.0], dtype=torch.float32, device=self.device)
        ] * self.num_envs

        # keep check of rollout length per environment
        self.rollout_len = torch.zeros(
            (self.num_envs,), dtype=torch.int32, device=self.device
        )

        # replay buffer
        self.obs_buf = torch.zeros(
            (self.steps_num, self.num_envs, self.num_obs),
            dtype=torch.float32,
            device=self.device,
        )
        self.rew_buf = torch.zeros(
            (self.steps_num, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.done_mask = torch.zeros(
            (self.steps_num, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.next_values = torch.zeros(
            (self.steps_num, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.target_values = torch.zeros(
            (self.steps_num, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.ret = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)

        # for kl divergence computing
        self.old_mus = torch.zeros(
            (self.steps_num, self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.old_sigmas = torch.zeros(
            (self.steps_num, self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.mus = torch.zeros(
            (self.steps_num, self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.sigmas = torch.zeros(
            (self.steps_num, self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # loss variables
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_discounted_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_gamma = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_length = torch.zeros(self.num_envs, dtype=int)
        self.done_buf = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.value_loss = np.inf
        self.jacobians = []
        self.truncations = []
        self.contact_changes = []
        self.early_stops = []
        self.episode_ends = []

        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.score_keys = cfg["params"]["config"].get("score_keys", [])
        self.episode_scores_meter_map = {
            key + "_final": AverageMeter(1, 100).to(self.device)
            for key in self.score_keys
        }

        # timer
        self.time_report = TimeReport()

    def compute_actor_loss(self, deterministic=False):
        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        actor_loss_terms = 0  # number of additions to actor_loss

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)

            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

        # fetch last observations
        obs = self.env.obs_buf
        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                self.obs_rms.update(obs)
            # normalize the current obs
            obs = obs_rms.normalize(obs)

        # accumulates all rollout lengths after they have been cut
        rollout_lens = []

        # Start short horizon rollout
        while actor_loss_terms < self.num_envs:
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[self.rollout_len] = obs.clone()

            # act in environment
            actions = self.actor(obs, deterministic=deterministic)
            obs, rew, term, trunc, info = self.env.step(torch.tanh(actions))

            # episode is done because we have reset the environment
            ep_done = trunc | term
            ep_done_env_ids = ep_done.nonzero(as_tuple=False).squeeze(-1).cpu()

            self.done_buf = self.done_buf | ep_done

            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    self.obs_rms.update(obs)
                # normalize the current obs
                obs = obs_rms.normalize(obs)

            if self.ret_rms is not None:
                # update ret rms
                with torch.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)

                rew = rew / torch.sqrt(ret_var + 1e-6)

            self.episode_length += 1
            self.rollout_len += 1

            # for logging
            self.early_stops.append(term.cpu().numpy())
            self.episode_ends.append(trunc.cpu().numpy())

            if "jacobian" in info:
                jac = info["jacobian"]  # shape NxSxA
                self.jacobians.append(jac)
                self.contact_changes.append(info["contacts_changed"].cpu().numpy())

                # do horizon trunction
                jac_norm = np.linalg.norm(jac, axis=(1, 2))
                contact_trunc = jac_norm > self.contact_th
                contact_trunc = tu.to_torch(contact_trunc, dtype=torch.int64)
                # ensure that we're not truncating envs before the minimum step size
                contact_trunc = contact_trunc & (self.rollout_len >= self.steps_min)
                # trunc = trunc | contact_trunc # NOTE: I don't think we need this anymore

            # for logging
            # self.truncations.append(trunc.cpu().numpy())

            real_obs = info["obs_before_reset"]
            # sanity check
            if (~torch.isfinite(real_obs)).sum() > 0:
                print("Got inf obs")
                raise ValueError

            if self.obs_rms is not None:
                real_obs = obs_rms.normalize(real_obs)

            next_values = self.target_critic(real_obs).squeeze(-1)

            # handle terminated environments
            term_env_ids = term.nonzero(as_tuple=False).squeeze(-1)
            for id in term_env_ids:
                next_values[id] = 0.0

            # sanity check
            if (next_values > 1e6).sum() > 0 or (next_values < -1e6).sum() > 0:
                print("next value error")
                raise ValueError

            self.rew_acc += self.gamma**self.rollout_len * rew

            # now merge truncation and termination into done
            cutoff = self.rollout_len >= self.steps_num
            if "jacobian" in info:
                cutoff = cutoff | contact_trunc
            # print("terminated", term.nonzero().flatten().tolist())
            # print("truncated", trunc.nonzero().flatten().tolist())
            # print("cutoff", cutoff.nonzero().flatten().tolist())
            done = term | trunc | cutoff
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            # terminate all done environments
            # TODO vectorize somehow
            for k in done_env_ids:
                actor_loss -= (
                    self.rew_acc[k] + self.gamma ** self.rollout_len[k] * next_values[k]
                ).sum()

            # keep count of number of loss terms we've added so far
            actor_loss_terms += done.sum().item()

            # clear up buffers
            for k in done_env_ids:
                self.rew_acc[k] = torch.zeros_like(self.rew_acc[k])
            rollout_lens.extend(self.rollout_len[done_env_ids].tolist())
            self.rollout_len[done_env_ids] = 0

            # cut off gradients of all done envs
            # TODO do I still need this?
            # self.env.clear_grad_ids(done_env_ids)

            # get observations again since we need them detached
            obs = self.env.obs_buf
            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    self.obs_rms.update(obs)
                # normalize the current obs
                obs = obs_rms.normalize(obs)

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[self.rollout_len] = rew.clone()
                self.done_mask[self.rollout_len] = done.clone().to(torch.float32)
                self.next_values[self.rollout_len] = next_values.clone()

            # collect episode loss
            with torch.no_grad():
                self.episode_loss -= rew
                self.episode_discounted_loss -= self.episode_gamma * rew
                self.episode_gamma *= self.gamma
                if len(ep_done_env_ids) > 0:
                    self.episode_loss_meter.update(self.episode_loss[ep_done_env_ids])
                    self.episode_discounted_loss_meter.update(
                        self.episode_discounted_loss[ep_done_env_ids]
                    )
                    self.episode_length_meter.update(
                        self.episode_length[ep_done_env_ids]
                    )
                    for k, v in filter(lambda x: x[0] in self.score_keys, info.items()):
                        self.episode_scores_meter_map[k + "_final"].update(
                            v[ep_done_env_ids]
                        )
                    for ep_done_env_ids in ep_done_env_ids:
                        if (
                            self.episode_loss[ep_done_env_ids] > 1e6
                            or self.episode_loss[ep_done_env_ids] < -1e6
                        ):
                            print("ep loss error")
                            raise ValueError

                        self.episode_loss_his.append(
                            self.episode_loss[ep_done_env_ids].item()
                        )
                        self.episode_discounted_loss_his.append(
                            self.episode_discounted_loss[ep_done_env_ids].item()
                        )
                        self.episode_length_his.append(
                            self.episode_length[ep_done_env_ids].item()
                        )
                        self.episode_loss[ep_done_env_ids] = 0.0
                        self.episode_discounted_loss[ep_done_env_ids] = 0.0
                        self.episode_length[ep_done_env_ids] = 0
                        self.episode_gamma[ep_done_env_ids] = 1.0

        steps = np.sum(rollout_lens)
        actor_loss /= steps

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)

        self.actor_loss = actor_loss.detach().item()

        self.step_count += steps

        self.mean_horizon = np.mean(rollout_lens)

        if torch.all(self.done_buf):
            print("RESETTING ALL ENVS")
            # self.env.reset(force_reset=True)
            # self.env.reset(torch.arange(0, self.num_envs))
            self.env.initialize_trajectory()
            self.done_buf = torch.zeros(
                (self.num_envs,), dtype=bool, device=self.device
            )

            # technically reduces performance
            # self.rew_acc = [
            # torch.tensor([0.0], dtype=torch.float32, device=self.device)
            # ] * self.num_envs
            # self.rollout_len = torch.zeros_like(self.rollout_len)

        return actor_loss

    @torch.no_grad()
    def evaluate_policy(self, num_games, deterministic=False):
        episode_length_his = []
        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        episode_length = torch.zeros(self.num_envs, dtype=int)
        episode_gamma = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        episode_discounted_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        obs = self.env.reset()

        games_cnt = 0
        while games_cnt < num_games:
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)

            actions = self.actor(obs, deterministic=deterministic)

            obs, rew, term, trunc, _ = self.env.step(torch.tanh(actions), play=True)
            done = term | trunc

            episode_length += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            episode_loss -= rew
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print(
                        "loss = {:.2f}, len = {}".format(
                            episode_loss[done_env_id].item(),
                            episode_length[done_env_id],
                        )
                    )
                    episode_loss_his.append(episode_loss[done_env_id].item())
                    episode_discounted_loss_his.append(
                        episode_discounted_loss[done_env_id].item()
                    )
                    episode_length_his.append(episode_length[done_env_id].item())
                    episode_loss[done_env_id] = 0.0
                    episode_discounted_loss[done_env_id] = 0.0
                    episode_length[done_env_id] = 0
                    episode_gamma[done_env_id] = 1.0
                    games_cnt += 1

        mean_episode_length = np.mean(np.array(episode_length_his))
        mean_policy_loss = np.mean(np.array(episode_loss_his))
        mean_policy_discounted_loss = np.mean(np.array(episode_discounted_loss_his))

        return mean_policy_loss, mean_policy_discounted_loss, mean_episode_length

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == "one-step":
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == "td-lambda":
            Ai = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1.0 - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (
                    self.lam * self.gamma * Ai
                    + self.gamma * self.next_values[i]
                    + (1.0 - lam) / (1.0 - self.lam) * self.rew_buf[i]
                )
                Bi = (
                    self.gamma
                    * (
                        self.next_values[i] * self.done_mask[i]
                        + Bi * (1.0 - self.done_mask[i])
                    )
                    + self.rew_buf[i]
                )
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError

    def compute_critic_loss(self, batch_sample):
        predicted_values = self.critic(batch_sample["obs"]).squeeze(-1)
        target_values = batch_sample["target_values"]
        critic_loss = ((predicted_values - target_values) ** 2).mean()

        return critic_loss

    def initialize_env(self):
        self.env.clear_grad()
        self.env.reset()

    @torch.no_grad()
    def run(self, num_games):
        (
            mean_policy_loss,
            mean_policy_discounted_loss,
            mean_episode_length,
        ) = self.evaluate_policy(
            num_games=num_games, deterministic=not self.stochastic_evaluation
        )
        print_info(
            "mean episode loss = {}, mean discounted loss = {}, mean episode length = {}".format(
                mean_policy_loss, mean_policy_discounted_loss, mean_episode_length
            )
        )

    def train(self):
        self.start_time = time.time()

        # add timers
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("compute actor loss")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("backward simulation")
        self.time_report.add_timer("prepare critic dataset")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")

        self.time_report.start_timer("algorithm")

        # initializations
        self.initialize_env()
        self.episode_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_discounted_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_length = torch.zeros(self.num_envs, dtype=int)
        self.episode_gamma = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        def actor_closure():
            self.actor_optimizer.zero_grad()

            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")
            actor_loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")
            # need to retain the graph so that we can backprop through the reward
            actor_loss.backward(retain_graph=True)
            self.time_report.end_timer("backward simulation")

            with torch.no_grad():
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters())

                # sanity check
                if (
                    torch.isnan(self.grad_norm_before_clip)
                    or self.grad_norm_before_clip > 1000000.0
                ):
                    print("NaN gradient")
                    raise ValueError

            self.time_report.end_timer("compute actor loss")

            return actor_loss

        # main training process
        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == "linear":
                actor_lr = (1e-5 - self.actor_lr) * float(
                    epoch / self.max_epochs
                ) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = actor_lr
                lr = actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(
                    epoch / self.max_epochs
                ) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = critic_lr
            else:
                lr = self.actor_lr

            # clear buffers for critic
            self.obs_buf = torch.zeros_like(self.obs_buf)

            # train actor
            self.time_report.start_timer("actor training")
            self.actor_optimizer.step(actor_closure).detach().item()
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                rew_backup = self.rew_buf.clone()
                value_backup = self.next_values.clone()
                obs_backup = self.obs_buf.clone()

                # set all rewards and values that haven't been finished to 0
                for n in range(self.num_envs):
                    # TODO not sure if this would make the critic learn 0 values
                    if self.rollout_len[n] != 0:
                        self.rew_buf[-self.rollout_len[n] :, n] = 0.0
                        self.next_values[-self.rollout_len[n] :, n] = 0.0
                        self.obs_buf[-self.rollout_len[n] :, n, :] = torch.nan
                        # NOTE: nans equal invalid data in the dataset below

                self.compute_target_values()
                dataset = CriticDataset(
                    self.batch_size,
                    self.obs_buf,
                    self.target_values,
                    drop_last=False,
                )
                # reset buffers correctly for next iteration
                for n in range(self.num_envs):
                    if self.rollout_len[n] != 0:
                        self.rew_buf[: self.rollout_len[n], n] = rew_backup[
                            -self.rollout_len[n] :, n
                        ]
                        self.next_values[: self.rollout_len[n], n] = value_backup[
                            -self.rollout_len[n] :, n
                        ]
                        self.obs_buf[: self.rollout_len[n], n] = obs_backup[
                            -self.rollout_len[n] :, n
                        ]

            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.0
            for j in range(self.critic_iterations):
                total_critic_loss = 0.0
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad()
                    training_critic_loss = self.compute_critic_loss(batch_sample)
                    training_critic_loss.backward()

                    # ugly fix for simulation nan problem
                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    if self.truncate_grad:
                        clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                    self.critic_optimizer.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1

                self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
                print(
                    "value iter {}/{}, loss = {:7.6f}".format(
                        j + 1, self.critic_iterations, self.value_loss
                    ),
                    end="\r",
                )

            self.time_report.end_timer("critic training")

            self.iter_count += 1

            time_end_epoch = time.time()

            fps = self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch)

            # logging
            time_elapse = time.time() - self.start_time
            self.writer.add_scalar("lr/iter", lr, self.iter_count)
            self.writer.add_scalar("actor_loss/step", self.actor_loss, self.step_count)
            self.writer.add_scalar("actor_loss/iter", self.actor_loss, self.iter_count)
            self.writer.add_scalar("value_loss/step", self.value_loss, self.step_count)
            self.writer.add_scalar("value_loss/iter", self.value_loss, self.iter_count)
            self.writer.add_scalar(
                "rollout_len/iter", self.mean_horizon, self.iter_count
            )
            self.writer.add_scalar(
                "rollout_len/step", self.mean_horizon, self.step_count
            )
            self.writer.add_scalar("rollout_len/time", self.mean_horizon, time_elapse)
            self.writer.add_scalar("fps/iter", fps, self.iter_count)
            self.writer.add_scalar("fps/step", fps, self.step_count)
            self.writer.add_scalar("fps/time", fps, time_elapse)
            if len(self.episode_loss_his) > 0:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_loss = self.episode_loss_meter.get_mean()
                mean_policy_discounted_loss = (
                    self.episode_discounted_loss_meter.get_mean()
                )

                if mean_policy_loss < self.best_policy_loss:
                    print_info(
                        "save best policy with loss {:.2f}".format(mean_policy_loss)
                    )
                    self.save()
                    self.best_policy_loss = mean_policy_loss

                self.writer.add_scalar(
                    "policy_loss/step", mean_policy_loss, self.step_count
                )
                self.writer.add_scalar(
                    "policy_loss/time", mean_policy_loss, time_elapse
                )
                self.writer.add_scalar(
                    "policy_loss/iter", mean_policy_loss, self.iter_count
                )
                self.writer.add_scalar(
                    "rewards/step", -mean_policy_loss, self.step_count
                )
                self.writer.add_scalar("rewards/time", -mean_policy_loss, time_elapse)
                self.writer.add_scalar(
                    "rewards/iter", -mean_policy_loss, self.iter_count
                )
                if (
                    self.score_keys
                    and len(
                        self.episode_scores_meter_map[self.score_keys[0] + "_final"]
                    )
                    > 0
                ):
                    for score_key in self.score_keys:
                        score = self.episode_scores_meter_map[
                            score_key + "_final"
                        ].get_mean()
                        self.writer.add_scalar(
                            "scores/{}/iter".format(score_key), score, self.iter_count
                        )
                        self.writer.add_scalar(
                            "scores/{}/step".format(score_key), score, self.step_count
                        )
                        self.writer.add_scalar(
                            "scores/{}/time".format(score_key), score, time_elapse
                        )
                self.writer.add_scalar(
                    "policy_discounted_loss/step",
                    mean_policy_discounted_loss,
                    self.step_count,
                )
                self.writer.add_scalar(
                    "policy_discounted_loss/iter",
                    mean_policy_discounted_loss,
                    self.iter_count,
                )
                self.writer.add_scalar(
                    "best_policy_loss/step", self.best_policy_loss, self.step_count
                )
                self.writer.add_scalar(
                    "best_policy_loss/iter", self.best_policy_loss, self.iter_count
                )
                self.writer.add_scalar(
                    "episode_lengths/iter", mean_episode_length, self.iter_count
                )
                self.writer.add_scalar(
                    "episode_lengths/step", mean_episode_length, self.step_count
                )
                self.writer.add_scalar(
                    "episode_lengths/time", mean_episode_length, time_elapse
                )
                ac_stddev = self.actor.get_logstd().exp().mean().detach().cpu().item()
                self.writer.add_scalar("ac_std/iter", ac_stddev, self.iter_count)
                self.writer.add_scalar("ac_std/step", ac_stddev, self.step_count)
                self.writer.add_scalar("ac_std/time", ac_stddev, time_elapse)
                self.writer.add_scalar(
                    "actor_grad_norm/iter", self.grad_norm_before_clip, self.iter_count
                )
                self.writer.add_scalar(
                    "actor_grad_norm/step", self.grad_norm_before_clip, self.step_count
                )
            else:
                mean_policy_loss = np.inf
                mean_policy_discounted_loss = np.inf
                mean_episode_length = 0

            np.savez(
                open(os.path.join(self.log_dir, "jacobians.npz"), "wb"),
                jacobians=self.jacobians,
                contact_changes=self.contact_changes,
                truncations=self.truncations,
                early_stops=self.early_stops,
                episode_ends=self.episode_ends,
            )

            print(
                "iter {:}/{:}, ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, avg rollout {:.1f}, total steps {:}, fps {:.2f}, value loss {:.2f}, grad norm before clip {:.2f}, grad norm after clip {:.2f}".format(
                    self.iter_count,
                    self.max_epochs,
                    mean_policy_loss,
                    mean_policy_discounted_loss,
                    mean_episode_length,
                    self.mean_horizon,
                    self.step_count,
                    fps,
                    self.value_loss,
                    self.grad_norm_before_clip,
                    self.grad_norm_after_clip,
                )
            )

            self.writer.flush()

            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(
                    self.name
                    + "policy_iter{}_reward{:.3f}".format(
                        self.iter_count, -mean_policy_loss
                    )
                )

            # update target critic
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(
                    self.critic.parameters(), self.target_critic.parameters()
                ):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1.0 - alpha) * param.data)

        self.time_report.end_timer("algorithm")

        self.time_report.report()

        self.save("final_policy")

        # save reward/length history
        self.episode_loss_his = np.array(self.episode_loss_his)
        self.episode_discounted_loss_his = np.array(self.episode_discounted_loss_his)
        self.episode_length_his = np.array(self.episode_length_his)
        np.save(
            open(os.path.join(self.log_dir, "episode_loss_his.npy"), "wb"),
            self.episode_loss_his,
        )
        np.save(
            open(os.path.join(self.log_dir, "episode_discounted_loss_his.npy"), "wb"),
            self.episode_discounted_loss_his,
        )
        np.save(
            open(os.path.join(self.log_dir, "episode_length_his.npy"), "wb"),
            self.episode_length_his,
        )

        # evaluate the final policy's performance
        self.run(self.eval_runs)

        self.close()

    def play(self, cfg):
        self.load(cfg["params"]["general"]["checkpoint"])
        self.run(cfg["params"]["config"]["player"]["games_num"])

    def save(self, filename=None):
        if filename is None:
            filename = "best_policy"
        torch.save(
            [self.actor, self.critic, self.target_critic, self.obs_rms, self.ret_rms],
            os.path.join(self.log_dir, "{}.pt".format(filename)),
        )

    def load(self, path):
        print("Loading policy from", path)
        checkpoint = torch.load(path)
        self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.target_critic = checkpoint[2].to(self.device)
        self.obs_rms = checkpoint[3].to(self.device)
        self.ret_rms = (
            checkpoint[4].to(self.device)
            if checkpoint[4] is not None
            else checkpoint[4]
        )

    def close(self):
        self.writer.close()
