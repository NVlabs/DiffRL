# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Adaptive Horizon Actor Critic (AHAC) is an alteration of SHAC. Instead
# of rolling out all envs in parallel for a fixed horizon, this attempts
# to rollout each env until it needs to be truncated. This can be viewed
# as an asynchronus rollout scheme where the gradients flowing back from
# each env are truncated independently from the others.

# Learning horizons inspired by Lagrange dual optimization
# In comparison to SHAC4, here we learn without contact thresholds

import sys, os

from torch.nn.utils.clip_grad import clip_grad_norm_

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import time
import copy
from tensorboardX import SummaryWriter
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Optional, List, Tuple
from collections import deque

from shac.utils.common import *
import shac.utils.torch_utils as tu
from shac.utils.running_mean_std import RunningMeanStd
from shac.utils.dataset import CriticDataset, QCriticDataset
from shac.utils.time_report import TimeReport
from shac.utils.average_meter import AverageMeter


class AHAC5:
    def __init__(
        self,
        env_config: DictConfig,
        actor_config: DictConfig,
        critic_config: DictConfig,
        steps_min: int,  # minimum horizon
        steps_max: int,  # maximum horizon
        max_epochs: int,  # number of short rollouts to do (i.e. epochs)
        train: bool,  # if False, we only eval the policy
        logdir: str,
        grad_norm: Optional[float] = None,  # clip actor and ciritc grad norms
        critic_grad_norm: Optional[float] = None,
        # contact_threshold: float = 1e9,  # for cutting horizons
        accumulate_jacobians: bool = False,  # if true clip gradients by accumulation
        actor_lr: float = 2e-3,
        critic_lr: float = 2e-3,
        h_lr: float = 2e-3,
        betas: Tuple[float, float] = (0.7, 0.95),
        lr_schedule: str = "linear",
        gamma: float = 0.99,
        lam: float = 0.95,
        rew_scale: float = 1.0,
        obs_rms: bool = False,
        ret_rms: bool = False,
        critic_iterations: Optional[int] = None,  # if None, we do early stop
        critic_batches: int = 4,
        critic_method: str = "one-step",
        save_interval: int = 500,  # how often to save policy
        stochastic_eval: bool = False,  # Whether to use stochastic actor in eval
        score_keys: List[str] = [],
        eval_runs: int = 12,
        log_jacobians: bool = False,  # expensive and messes up wandb
        device: str = "cuda",
    ):
        # sanity check parameters
        assert steps_max > steps_min > 0
        assert max_epochs > 0
        assert actor_lr >= 0
        assert critic_lr >= 0
        assert h_lr >= 0
        assert lr_schedule in ["linear", "constant"]
        assert 0 < gamma <= 1
        assert 0 < lam <= 1
        assert rew_scale > 0.0
        assert critic_iterations is None or critic_iterations > 0
        assert critic_batches > 0
        assert critic_method in ["one-step", "td-lambda"]
        assert save_interval > 0
        assert eval_runs >= 0

        # Create environment
        self.env = instantiate(env_config, logdir=logdir)
        print("num_envs = ", self.env.num_envs)
        print("num_actions = ", self.env.num_actions)
        print("num_obs = ", self.env.num_obs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length
        self.device = torch.device(device)

        self.steps_min = steps_min
        self.steps_max = steps_max
        self.H = torch.tensor(steps_min, dtype=torch.float32, device=self.device)
        self.lambd = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        self.max_epochs = max_epochs
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.lr_schedule = lr_schedule

        self.gamma = gamma
        self.lam = lam
        self.rew_scale = rew_scale

        self.critic_method = critic_method
        self.critic_iterations = critic_iterations
        self.critic_batches = critic_batches
        self.critic_batch_size = self.num_envs * self.steps_max // critic_batches

        self.obs_rms = None
        if obs_rms:
            self.obs_rms = RunningMeanStd(shape=(self.num_obs), device=self.device)

        self.ret_rms = None
        if ret_rms:
            self.ret_rms = RunningMeanStd(shape=(), device=self.device)

        env_name = self.env.__class__.__name__
        self.name = self.__class__.__name__ + "_" + env_name

        self.grad_norm = grad_norm
        self.critic_grad_norm = critic_grad_norm
        self.stochastic_evaluation = stochastic_eval
        self.save_interval = save_interval

        if train:
            self.log_dir = logdir
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.log_dir, "log"))

        # Create actor and critic
        self.actor = instantiate(
            actor_config,
            obs_dim=self.num_obs,
            action_dim=self.num_actions,
            device=self.device,
        )

        self.critic = instantiate(
            critic_config,
            obs_dim=self.num_obs,
            device=self.device,
        )

        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())

        # for logging purposes
        self.jac_buffer = []
        self.jacs = []
        # self.cfs = []
        self.early_terms = []
        self.conatct_truncs = []
        self.horizon_truncs = []
        self.episode_ends = []
        self.episode = 0

        if train:
            self.save("init_policy")

        # initialize optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            self.actor_lr,
            betas,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            self.critic_lr,
            betas,
        )
        self.h_lr = h_lr
        self.lambd_lr = 1e-4

        # NOTE: Don't need for single env
        # accumulate rewards for each environment
        # TODO make this vectorized somehow
        # self.rew_acc = [
        #     torch.tensor([0.0], dtype=torch.float32, device=self.device)
        # ] * self.num_envs

        # # keep check of rollout length per environment
        # self.rollout_len = torch.zeros(
        #     (self.num_envs,), dtype=torch.int32, device=self.device
        # )

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
        self.cfs = torch.zeros(
            (self.steps_num, self.num_envs), dtype=torch.float32, device=self.device
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
        # NOTE: do not need for single env
        # self.episode_length = torch.zeros(self.num_envs, dtype=int, device=self.device)
        # self.done_buf = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.value_loss = np.inf
        self.grad_norm_before_clip = np.inf
        self.grad_norm_after_clip = np.inf
        self.early_termination = 0
        self.episode_end = 0
        self.contact_trunc = 0
        self.horizon_trunc = 0
        self.acc_jacobians = accumulate_jacobians
        self.log_jacobians = log_jacobians
        self.eval_runs = eval_runs
        self.last_steps = 0
        self.last_log_steps = 0

        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.horizon_length_meter = AverageMeter(1, 100).to(self.device)
        self.score_keys = score_keys
        self.episode_scores_meter_map = {
            key + "_final": AverageMeter(1, 100).to(self.device)
            for key in self.score_keys
        }

        # temporary load policy
        # path = "/home/ignat/git/SHAC/scripts/multirun/2023-08-24/18-43-28/14/logs/best_policy.pt"
        # self.load(path, actor=False)

        # timer
        self.time_report = TimeReport()

    @property
    def mean_horizon(self):
        return self.horizon_length_meter.get_mean()

    @property
    def steps_num(self):
        return round(self.H.item())

    def compute_actor_loss(self, deterministic=False):
        rew_acc = torch.zeros(
            (self.steps_num + 1, self.num_envs), dtype=torch.float32, device=self.device
        )
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        next_values = torch.zeros(
            (self.steps_num + 1, self.num_envs), dtype=torch.float32, device=self.device
        )

        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)

            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

        # initialize trajectory to cut off gradients between episodes.
        obs = self.env.initialize_trajectory()
        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                self.obs_rms.update(obs)
            # normalize the current obs
            obs = obs_rms.normalize(obs)

        # keeps track of the current length of the rollout
        rollout_len = torch.zeros((self.num_envs,), device=self.device)
        # Start short horizon rollout
        for i in range(self.steps_num):
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[i] = obs.clone()

            # act in environment
            actions = self.actor(obs, deterministic=deterministic)
            obs, rew, done, info = self.env.step(torch.tanh(actions))
            term = info["termination"]
            trunc = info["truncation"]

            with torch.no_grad():
                raw_rew = rew.clone()

            # scale the reward
            rew = rew * self.rew_scale

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
            rollout_len += 1

            # contact truncation
            # defaults to jacobian truncation if they are available, otherwise
            # uses contact forces since they are always available
            cfs = info["contact_forces"]
            acc = info["accelerations"]
            cfs_normalised = torch.where(acc != 0.0, cfs / acc, torch.zeros_like(cfs))
            self.cfs[i] = torch.norm(cfs_normalised, dim=(1, 2))

            if self.log_jacobians:
                jac_norm = (
                    np.linalg.norm(info["jacobian"]) if "jacobian" in info else None
                )
                k = self.step_count + int(torch.sum(rollout_len).item())
                if jac_norm:
                    self.writer.add_scalar("jacobian", jac_norm, k)
                self.writer.add_scalar("contact_forces", cfs_normalised, k)

            real_obs = info["obs_before_reset"]
            # sanity check
            if (~torch.isfinite(real_obs)).sum() > 0:
                print("Got inf obs")
                # raise ValueError # it's ok to have this for humanoid

            if self.obs_rms is not None:
                real_obs = obs_rms.normalize(real_obs)

            next_values[i + 1] = self.critic(real_obs).squeeze(-1)

            # handle terminated environments which stopped for some bad reason
            # since the reason is bad we set their value to 0
            term_env_ids = term.nonzero(as_tuple=False).squeeze(-1)
            for id in term_env_ids:
                next_values[i + 1, id] = 0.0

            # sanity check
            if (next_values > 1e6).sum() > 0 or (next_values < -1e6).sum() > 0:
                print("next value error")
                raise ValueError

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            self.early_terms.append(torch.all(term).item())
            self.horizon_truncs.append(i == self.steps_num - 1)
            self.episode_ends.append(torch.all(trunc).item())

            done = term | trunc
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            self.early_termination += torch.sum(term).item()
            self.episode_end += torch.sum(trunc).item()

            if i < self.steps_num - 1:
                # first terminate all rollouts which are 'done'
                retrn = (
                    -rew_acc[i + 1, done_env_ids]
                    - self.gamma
                    * gamma[done_env_ids]
                    * next_values[i + 1, done_env_ids]
                )
                actor_loss += retrn.sum()
                with torch.no_grad():
                    self.ret[done_env_ids] += retrn
            else:
                # terminate all envs because we reached the end of our rollout
                retrn = -rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]
                actor_loss += retrn.sum()
                with torch.no_grad():
                    self.ret += retrn

            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.0
            rew_acc[i + 1, done_env_ids] = 0.0

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.steps_num - 1:
                    self.done_mask[i] = done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.0
                self.next_values[i] = next_values[i + 1].clone()

            # collect episode loss
            with torch.no_grad():
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                if len(done_env_ids) > 0:
                    self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                    self.episode_discounted_loss_meter.update(
                        self.episode_discounted_loss[done_env_ids]
                    )
                    self.episode_length_meter.update(self.episode_length[done_env_ids])
                    self.horizon_length_meter.update(rollout_len[done_env_ids])
                    rollout_len[done_env_ids] = 0
                    for k, v in filter(lambda x: x[0] in self.score_keys, info.items()):
                        self.episode_scores_meter_map[k + "_final"].update(
                            v[done_env_ids]
                        )
                    for id in done_env_ids:
                        if self.episode_loss[id] > 1e6 or self.episode_loss[id] < -1e6:
                            print("ep loss error")
                            raise ValueError

                        self.episode_loss_his.append(self.episode_loss[id].item())
                        self.episode_discounted_loss_his.append(
                            self.episode_discounted_loss[id].item()
                        )
                        self.episode_length_his.append(self.episode_length[id].item())
                        self.episode_loss[id] = 0.0
                        self.episode_discounted_loss[id] = 0.0
                        self.episode_length[id] = 0
                        self.episode_gamma[id] = 1.0

        self.horizon_length_meter.update(rollout_len)

        actor_loss /= self.steps_num * self.num_envs

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)

        self.actor_loss = actor_loss.detach().item()

        self.step_count += self.steps_num * self.num_envs

        if (
            self.log_jacobians
            and self.step_count - self.last_log_steps > 1000 * self.num_envs
        ):
            np.savez(
                os.path.join(self.log_dir, f"truncation_analysis_{self.episode}"),
                contact_forces=self.cfs,
                early_termination=self.early_terms,
                horizon_truncation=self.horizon_truncs,
                episode_ends=self.episode_ends,
            )
            # self.cfs = []
            self.early_terms = []
            self.horizon_truncs = []
            self.episode_ends = []
            self.episode += 1
            self.last_log_steps = self.step_count

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

            obs, rew, done, _ = self.env.step(torch.tanh(actions))

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
        predicted_values = self.critic.predict(batch_sample["obs"]).squeeze(-2)
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
        self.episode_length = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
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
            actor_loss.backward()
            self.time_report.end_timer("backward simulation")

            with torch.no_grad():
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                if self.grad_norm:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters())

                # sanity check
                if (
                    torch.isnan(self.grad_norm_before_clip)
                    or self.grad_norm_before_clip > 1e6
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

                h_lr = (1e-5 - self.h_lr) * epoch / self.max_epochs + self.h_lr
            else:
                lr = self.actor_lr
                h_lr = self.h_lr

            # train actor
            self.time_report.start_timer("actor training")
            self.actor_optimizer.step(actor_closure)
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                critic_batch_size = (
                    self.num_envs * self.steps_num // self.critic_batches
                )

                dataset = CriticDataset(
                    critic_batch_size,
                    self.obs_buf,
                    self.target_values,
                    drop_last=False,
                )
            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.0
            last_losses = deque(maxlen=5)
            iterations = self.critic_iterations if self.critic_iterations else 64
            for j in range(iterations):
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

                    if self.critic_grad_norm:
                        clip_grad_norm_(self.critic.parameters(), self.critic_grad_norm)

                    self.critic_optimizer.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1

                total_critic_loss /= batch_cnt
                if self.critic_iterations is None and len(last_losses) == 5:
                    diff = abs(np.diff(last_losses).mean())
                    if diff < 2e-1:
                        iterations = j + 1
                        break
                last_losses.append(total_critic_loss.item())

                self.value_loss = total_critic_loss
                print(
                    "value iter {}/{}, loss = {:7.6f}".format(
                        j + 1, iterations, self.value_loss
                    ),
                    end="\r",
                )

            self.time_report.end_timer("critic training")

            last_steps = self.steps_num

            # Train horizon
            h_grad = -self.ret / self.H - self.lambd * self.H
            self.H += h_lr * h_grad.mean()
            self.H = torch.clip(self.H, self.steps_min, self.steps_max)

            # train lambda
            self.lambd += self.lambd_lr * (self.H - self.steps_min)

            # printing for debugging purposes
            rew_w = (-self.ret / self.H).mean().item()
            constraint_w = (-self.lambd * self.H).item()
            print(f"weights reward {rew_w:.2f}  constraint {constraint_w:.2f}")
            print(f"H={self.H.item():.2f}, lambda={self.lambd.item():.2f}")

            # reset buffers correctly for next iteration
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
            self.ret = torch.zeros(
                (self.num_envs), dtype=torch.float32, device=self.device
            )
            self.cfs = torch.zeros(
                (self.steps_num, self.num_envs), dtype=torch.float32, device=self.device
            )

            self.iter_count += 1

            time_end_epoch = time.time()

            fps = last_steps * self.num_envs / (time_end_epoch - time_start_epoch)

            # logging
            self.log_scalar("lr", lr)
            self.log_scalar("actor_loss", self.actor_loss)
            self.log_scalar("value_loss", self.value_loss)
            self.log_scalar("rollout_len", self.mean_horizon)
            self.log_scalar("fps", fps)
            self.log_scalar("critic_iterations", iterations)

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

                self.log_scalar("policy_loss", mean_policy_loss)
                self.log_scalar("rewards", -mean_policy_loss)

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
                        self.log_scalar(f"scores/{score_key}", score)

                self.log_scalar("policy_discounted_loss", mean_policy_discounted_loss)
                self.log_scalar("best_policy_loss", self.best_policy_loss)
                self.log_scalar("episode_lengths", mean_episode_length)
                ac_stddev = self.actor.get_logstd().exp().mean().detach().cpu().item()
                self.log_scalar("ac_std", ac_stddev)
                self.log_scalar("actor_grad_norm", self.grad_norm_before_clip)
                self.log_scalar("episode_end", self.episode_end)
                self.log_scalar("early_termination", self.early_termination)
                self.log_scalar("horizon_trunc", self.horizon_trunc)
                self.log_scalar("contact_trunc", self.contact_trunc)
            else:
                mean_policy_loss = np.inf
                mean_policy_discounted_loss = np.inf
                mean_episode_length = 0

            print(
                "iter {:}/{:}, ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, H {:.2f}, avg rollout {:.1f}, total steps {:}, fps {:.2f}, value loss {:.2f}, contact/horizon/end {:}/{:}/{:}, grad norm before/after clip {:.2f}/{:.2f}".format(
                    self.iter_count,
                    self.max_epochs,
                    mean_policy_loss,
                    mean_policy_discounted_loss,
                    mean_episode_length,
                    self.H.item(),
                    self.mean_horizon,
                    self.step_count,
                    fps,
                    self.value_loss,
                    self.contact_trunc,
                    self.horizon_trunc,
                    self.episode_end,
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

        self.time_report.end_timer("algorithm")

        self.time_report.report()

        self.save("final_policy")

        # save reward/length history
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

    def save(self, filename=None):
        if filename is None:
            filename = "best_policy"
        torch.save(
            [self.actor, self.critic, self.obs_rms, self.ret_rms],
            os.path.join(self.log_dir, "{}.pt".format(filename)),
        )

    def load(self, path, actor=True):
        print("Loading policy from", path)
        checkpoint = torch.load(path)
        if actor:
            self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.obs_rms = checkpoint[2].to(self.device)
        self.ret_rms = (
            checkpoint[3].to(self.device)
            if checkpoint[3] is not None
            else checkpoint[3]
        )

    def log_scalar(self, scalar, value):
        """Helper method for consistent logging"""
        self.writer.add_scalar(f"{scalar}", value, self.iter_count)

    def close(self):
        self.writer.close()
