# Added option to resume training and loa Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import math
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os
import sys
import time

import yaml

from rl_games.common import schedulers
from rl_games.algos_torch import torch_ext
from shac.utils.average_meter import AverageMeter
from shac.utils.common import *
from shac.utils.dataset import CriticDataset
from shac.utils.running_mean_std import RunningMeanStd
from shac.utils.time_report import TimeReport
import shac.utils.torch_utils as tu
from tensorboardX import SummaryWriter
import torch.distributed as dist
from torch.nn.utils.clip_grad import clip_grad_norm_

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)


class SHAC:
    def __init__(self, cfg: dict):
        seeding(cfg["general"]["seed"])
        if "diff_env" not in cfg.env.config:
            self.env = instantiate(cfg.env.config)
        else:
            self.env = instantiate(cfg.env.config.diff_env)

        print("num_envs = ", self.env.num_envs)
        print("num_actions = ", self.env.num_actions)
        print("num_obs = ", self.env.num_obs)

        self.multi_gpu = cfg["alg"]["params"]["config"].get("multi_gpu", False)
        self.rank = 0
        self.rank_size = 1

        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)

            self.device_name = "cuda:" + str(self.rank)
            cfg["general"]["device"] = self.device_name
            if self.rank != 0:
                cfg["alg"]["params"]["config"]["print_stats"] = False
                # cfg["params"]["config"]["lr_schedule"] = None

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length
        self.device = cfg["general"]["device"]

        self.gamma = cfg["alg"]["params"]["config"].get("gamma", 0.99)

        self.critic_method = cfg["alg"]["params"]["config"].get(
            "critic_method", "one-step"
        )  # ['one-step', 'td-lambda']
        if self.critic_method == "td-lambda":
            self.lam = cfg["alg"]["params"]["config"].get("lambda", 0.95)

        self.steps_num = cfg["alg"]["params"]["config"]["steps_num"]
        self.max_epochs = cfg["alg"]["params"]["config"]["max_epochs"]
        self.actor_lr = float(cfg["alg"]["params"]["default_actor_opt"]["lr"])
        self.critic_lr = float(cfg["alg"]["params"]["default_critic_opt"]["lr"])
        self.lr_schedule = cfg["alg"]["params"]["config"].get("lr_schedule", "linear")

        self.is_adaptive_lr = self.lr_schedule == "adaptive"
        self.is_linear_lr = self.lr_schedule == "linear"

        if self.is_adaptive_lr:
            self.scheduler = instantiate(
                cfg["alg"]["params"]["default_adaptive_scheduler"]
            )
        elif self.is_linear_lr:
            self.scheduler = instantiate(
                cfg["alg"]["params"]["default_linear_scheduler"]
            )
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.target_critic_alpha = cfg["alg"]["params"]["config"].get(
            "target_critic_alpha", 0.4
        )

        self._obs_rms = None
        self.curr_epoch = 0
        self.sub_traj_per_epoch = math.ceil(self.max_episode_length / self.steps_num)
        # number of epochs of no improvement for early stopping
        self.early_stopping_patience = cfg["alg"]["params"]["config"].get(
            "early_stopping_patience", self.max_epochs
        )
        if cfg["alg"]["params"]["config"].get("obs_rms", False):
            # generate obs_rms for each subtrajectory
            self._obs_rms = [
                RunningMeanStd(shape=(self.num_obs), device=self.device)
                # for _ in range(self.sub_traj_per_epoch)
            ]

        self.ret_rms = None
        if cfg["alg"]["params"]["config"].get("ret_rms", False):
            self.ret_rms = RunningMeanStd(shape=(), device=self.device)

        self.rew_scale = cfg["alg"]["params"]["config"].get("rew_scale", 1.0)

        self.critic_iterations = cfg["alg"]["params"]["config"].get(
            "critic_iterations", 16
        )
        self.num_batch = cfg["alg"]["params"]["config"].get("num_batch", 4)
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.name = cfg["alg"]["params"]["config"]

        self.truncate_grad = cfg["alg"]["params"]["config"]["truncate_grads"]
        self.grad_norm = cfg["alg"]["params"]["config"]["grad_norm"]

        if cfg["general"]["train"]:
            self.log_dir = cfg["general"]["logdir"]
            if not self.multi_gpu or self.rank == 0:
                os.makedirs(self.log_dir, exist_ok=True)
                # save config
                save_cfg = OmegaConf.to_yaml(cfg)
                yaml.dump(save_cfg, open(os.path.join(self.log_dir, "cfg.yaml"), "w"))
                self.writer = SummaryWriter(os.path.join(self.log_dir, "log"))
            # save interval
            self.save_interval = cfg["alg"]["params"]["config"].get(
                "save_interval", 500
            )
            # stochastic inference
            self.stochastic_evaluation = True
        else:
            self.stochastic_evaluation = not (
                cfg["params"]["config"]["player"].get("determenistic", False)
                or cfg["params"]["config"]["player"].get("deterministic", False)
            )
            self.steps_num = self.env.episode_length

        # create actor critic network
        self.actor = instantiate(
            cfg["alg"]["params"]["network"]["actor"],
            num_obs=self.num_obs,
            num_actions=self.num_actions,
            device=self.device,
        )
        self.critic = instantiate(
            cfg["alg"]["params"]["network"]["critic"],
            num_obs=self.num_obs,
            num_actions=self.num_actions,
            device=self.device,
        )
        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.target_critic = copy.deepcopy(self.critic)

        # initialize optimizer
        self.actor_optimizer = instantiate(
            cfg["alg"]["params"]["config"]["actor_optimizer"],
            params=self.actor.parameters(),
        )
        self.critic_optimizer = instantiate(
            cfg["alg"]["params"]["config"]["critic_optimizer"],
            params=self.critic.parameters(),
        )

        if cfg["general"]["train"]:
            self.save("init_policy")

        self.mixed_precision = cfg["alg"]["params"]["config"].get(
            "mixed_precision", False
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

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
        self.best_policy_loss = np.inf
        self.best_policy_epoch = 0
        self.actor_loss = np.inf
        self.value_loss = np.inf

        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.score_keys = cfg["alg"]["params"]["config"].get("score_keys", [])
        self.episode_scores_meter_map = {
            key + "_final": AverageMeter(1, 100).to(self.device)
            for key in self.score_keys
        }

        # timer
        self.time_report = TimeReport()

    @property
    def obs_rms(self):
        if self._obs_rms is None:
            return self._obs_rms
        return self._obs_rms[0]  # self.curr_epoch % self.sub_traj_per_epoch]

    def compute_actor_loss(self, deterministic=False):
        rew_acc = torch.zeros(
            (self.steps_num + 1, self.num_envs), dtype=torch.float32, device=self.device
        )
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        next_values = torch.zeros(
            (self.steps_num + 1, self.num_envs), dtype=torch.float32, device=self.device
        )
        next_values_model_free = torch.zeros(
            (self.steps_num + 1, self.num_envs), dtype=torch.float32, device=self.device
        )

        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        actor_model_free_loss = torch.tensor(
            0.0, dtype=torch.float32, device=self.device
        )

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
        for i in range(self.steps_num):
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[i] = obs.clone()

            actions = self.actor(obs, deterministic=deterministic)
            self.mus[i, :], _, self.sigmas[i, :] = self.actor.forward_with_dist(
                obs, deterministic=False
            )

            obs, rew, done, extra_info = self.env.step(torch.tanh(actions))

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

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            next_values[i + 1] = torch.minimum(
                self.critic(obs).squeeze(-1), self.target_critic(obs).squeeze(-1)
            )
            next_values_model_free[i + 1] = torch.minimum(
                self.critic(obs.requires_grad_(False)).squeeze(-1),
                self.target_critic(obs.require_grad_(False)).squeeze(-1),
            )

            for id in done_env_ids:
                if (
                    torch.isnan(extra_info["obs_before_reset"][id]).sum() > 0
                    or torch.isinf(extra_info["obs_before_reset"][id]).sum() > 0
                    or (torch.abs(extra_info["obs_before_reset"][id]) > 1e6).sum() > 0
                ):  # ugly fix for nan values
                    next_values[i + 1, id] = 0.0
                elif (
                    self.episode_length[id] < self.max_episode_length
                ):  # early termination
                    next_values[i + 1, id] = 0.0
                else:  # otherwise, use terminal value critic to estimate the long-term performance
                    if self.obs_rms is not None:
                        real_obs = obs_rms.normalize(extra_info["obs_before_reset"][id])
                    else:
                        real_obs = extra_info["obs_before_reset"][id]
                    next_values[i + 1, id] = torch.minimum(
                        self.critic(real_obs).squeeze(-1),
                        self.target_critic(real_obs).squeeze(-1),
                    )

            if (next_values[i + 1] > 1e6).sum() > 0 or (
                next_values[i + 1] < -1e6
            ).sum() > 0:
                print("next value error")
                if self.multi_gpu:
                    dist.destroy_process_group()
                raise ValueError

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            if i < self.steps_num - 1:
                actor_loss += (
                    -rew_acc[i + 1, done_env_ids]
                    - self.gamma
                    * gamma[done_env_ids]
                    * next_values[i + 1, done_env_ids]
                ).sum()
                actor_model_free_loss += (
                    -rew_acc[i + 1, done_env_ids].detach()
                    - self.gamma
                    * gamma[done_env_ids]
                    * next_values_model_free[i + 1, done_env_ids]
                ).sum()
            else:
                # terminate all envs at the end of optimization iteration
                actor_loss += (
                    -rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]
                ).sum()
                actor_model_free_loss += (
                    -rew_acc[i + 1, :].detach()
                    - self.gamma * gamma * next_values_model_free[i + 1, :]
                ).sum()

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
                    for k, v in filter(
                        lambda x: x[0] in self.score_keys, extra_info.items()
                    ):
                        self.episode_scores_meter_map[k + "_final"].update(
                            v[done_env_ids]
                        )
                    for done_env_id in done_env_ids:
                        if (
                            self.episode_loss[done_env_id] > 1e6
                            or self.episode_loss[done_env_id] < -1e6
                        ):
                            print("ep loss error")
                            if self.multi_gpu:
                                dist.destroy_process_group()
                            raise ValueError

                        self.episode_loss_his.append(
                            self.episode_loss[done_env_id].item()
                        )
                        self.episode_discounted_loss_his.append(
                            self.episode_discounted_loss[done_env_id].item()
                        )
                        self.episode_length_his.append(
                            self.episode_length[done_env_id].item()
                        )
                        self.episode_loss[done_env_id] = 0.0
                        self.episode_discounted_loss[done_env_id] = 0.0
                        self.episode_length[done_env_id] = 0
                        self.episode_gamma[done_env_id] = 1.0

        actor_loss /= self.steps_num * self.num_envs
        actor_model_free_loss /= self.steps_num * self.num_envs

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)
            actor_model_free_loss = actor_model_free_loss * torch.sqrt(ret_var + 1e-6)

        self.actor_loss = actor_loss.detach().cpu().item()
        self.actor_model_free_loss = actor_loss.detach().cpu().item()

        self.step_count += self.steps_num * self.num_envs

        return actor_loss, actor_model_free_loss

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
        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            if self.rank == 0:
                print("====================broadcasting parameters")
                print("====actor parameters")
            actor_params = [self.actor.state_dict()]
            dist.broadcast_object_list(actor_params, 0)
            self.actor.load_state_dict(actor_params[0])
            if self.rank == 0:
                print("====critic parameters")
            critic_params = [self.critic.state_dict()]
            dist.broadcast_object_list(critic_params, 0)
            self.critic.load_state_dict(critic_params[0])
            if self.rank == 0:
                print("done broadcasting parameters====================")

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
            self.actor_optimizer.zero_grad(set_to_none=True)

            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")
            # env_state = self.env.get_checkpoint()
            # TODO: use autoscaling for mixed precision

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                actor_loss, actor_model_free_loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")
            self.scaler.scale(actor_loss).backward()
            self.time_report.end_timer("backward simulation")

            with torch.no_grad():
                self.scaler.unscale_(self.actor_optimizer)
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                self.truncate_gradients_and_step(
                    self.actor.parameters(), self.actor_optimizer, unscale=False
                )
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters())

                # sanity check
                if (
                    torch.isnan(self.grad_norm_before_clip)
                    or self.grad_norm_before_clip > 1000000.0
                ):
                    print("shac training crashed due to unstable gradient")
                    # torch.save(env_state, os.path.join(self.log_dir, "bad_state.pt"))
                    print("NaN gradient")
                    if not self.multi_gpu or self.rank == 0:
                        self.save("crashed")
                    if self.multi_gpu:
                        dist.destroy_process_group()
                    raise ValueError

            self.time_report.end_timer("compute actor loss")

            return actor_loss

        actor_lr, critic_lr = self.actor_lr, self.critic_lr
        print("starting training: lr = {}, {}".format(actor_lr, critic_lr))
        start_epoch = self.curr_epoch
        # main training process
        for epoch in range(start_epoch, self.max_epochs):
            self.curr_epoch += 1
            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == "linear":
                if self.rank == 0:
                    actor_lr, _ = self.scheduler.update(
                        actor_lr, None, self.curr_epoch, None, None
                    )
                    critic_lr, _ = self.scheduler.update(
                        critic_lr, None, self.curr_epoch, None, None
                    )

                if self.multi_gpu:
                    lr_tensor = torch.tensor([actor_lr, critic_lr], device=self.device)
                    dist.broadcast(lr_tensor, 0)
                    actor_lr = lr_tensor[0].item()
                    critic_lr = lr_tensor[1].item()

                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = critic_lr

                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = actor_lr
                lr = actor_lr
            elif self.lr_schedule == "adaptive":
                av_kls = torch_ext.mean_list(ep_kls)
                if self.multi_gpu:
                    dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                    av_kls /= self.rank_size
                kl_dist = torch_ext.policy_kl(
                    self.mus.detach(),
                    self.sigmas.detach(),
                    self.old_mus,
                    self.old_sigmas,
                    reduce=True,
                )

            else:
                lr = self.actor_lr

            # train actor
            self.time_report.start_timer("actor training")
            actor_loss = actor_closure()
            # self.truncate_gradients_and_step(
            #     self.actor.parameters(), self.actor_optimizer, unscale=False
            # )
            # self.actor_optimizer.step(actor_closure).detach().item()
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(
                    self.batch_size, self.obs_buf, self.target_values, drop_last=False
                )
            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.0
            for j in range(self.critic_iterations):
                total_critic_loss = 0.0
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                        training_critic_loss = self.compute_critic_loss(batch_sample)
                    self.scaler.scale(training_critic_loss).backward()

                    # ugly fix for simulation nan problem
                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    self.truncate_gradients_and_step(
                        self.critic.parameters(), self.critic_optimizer
                    )

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
            should_exit = False

            # update target critic
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(
                    self.critic.parameters(), self.target_critic.parameters()
                ):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1.0 - alpha) * param.data)

            # skip logging if not the head node
            if self.rank != 0 and self.multi_gpu:
                continue

            time_elapse = time.time() - self.start_time
            self.writer.add_scalar("lr/iter", lr, self.iter_count)
            self.writer.add_scalar("actor_loss/step", self.actor_loss, self.step_count)
            self.writer.add_scalar("actor_loss/iter", self.actor_loss, self.iter_count)
            self.writer.add_scalar("value_loss/step", self.value_loss, self.step_count)
            self.writer.add_scalar("value_loss/iter", self.value_loss, self.iter_count)
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
                    self.best_policy_epoch = self.curr_epoch
                # number of episodes with no improvement
                else:
                    last_improved_ep = self.best_policy_epoch - self.curr_epoch
                    if last_improved_ep > self.early_stopping_patience:
                        should_exit = True

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
            else:
                mean_policy_loss = np.inf
                mean_policy_discounted_loss = np.inf
                mean_episode_length = 0

            self.writer.flush()

            print(
                "iter {}: ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, fps total {:.2f}, value loss {:.2f}, grad norm before clip {:.2f}, grad norm after clip {:.2f}".format(
                    self.iter_count,
                    mean_policy_loss,
                    mean_policy_discounted_loss,
                    mean_episode_length,
                    self.steps_num
                    * self.num_envs
                    * self.rank_size
                    / (time_end_epoch - time_start_epoch),
                    self.value_loss,
                    self.grad_norm_before_clip,
                    self.grad_norm_after_clip,
                )
            )
            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(
                    self.name
                    + "policy_iter{}_reward{:.3f}".format(
                        self.iter_count, -mean_policy_loss
                    )
                )

            if should_exit:
                break

        self.time_report.end_timer("algorithm")

        self.time_report.report()

        if self.rank == 0 or not self.multi_gpu:
            self.save("final_policy")
            # save reward/length history
            self.episode_loss_his = np.array(self.episode_loss_his)
            self.episode_discounted_loss_his = np.array(
                self.episode_discounted_loss_his
            )
            self.episode_length_his = np.array(self.episode_length_his)
            np.save(
                open(os.path.join(self.log_dir, "episode_loss_his.npy"), "wb"),
                self.episode_loss_his,
            )
            np.save(
                open(
                    os.path.join(self.log_dir, "episode_discounted_loss_his.npy"), "wb"
                ),
                self.episode_discounted_loss_his,
            )
            np.save(
                open(os.path.join(self.log_dir, "episode_length_his.npy"), "wb"),
                self.episode_length_his,
            )

            # evaluate the final policy's performance
            self.run(self.num_envs)
            self.close()

        if self.multi_gpu and should_exit:
            dist.destroy_process_group()

    def truncate_gradients_and_step(self, parameters, optimizer, unscale=True):
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in parameters:
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))
            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in parameters:
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(
                            param.grad.data
                        )
                        / self.rank_size
                    )
                    offset += param.numel()

        if self.truncate_grad:
            if unscale:
                self.scaler.unscale_(optimizer)
            clip_grad_norm_(parameters, self.grad_norm)

        self.scaler.step(optimizer)
        self.scaler.update()

    def play(self, cfg):
        self.load(cfg["params"]["general"]["checkpoint"], cfg)
        self.run(cfg["params"]["config"]["player"]["games_num"])

    def save(self, filename=None):
        if filename is None:
            filename = "best_policy"
        torch.save(
            [
                self.actor.state_dict(),
                self.critic.state_dict(),
                self.target_critic.state_dict(),
                self._obs_rms,
                self.ret_rms,
                self.actor_optimizer.state_dict(),
                self.critic_optimizer.state_dict(),
            ],
            os.path.join(self.log_dir, "{}.pt".format(filename)),
        )

    def load(self, path, cfg, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        if isinstance(checkpoint[0], dict):
            self.actor.load_state_dict(checkpoint[0])
        else:
            self.actor = checkpoint[0].to(self.device)
        if isinstance(checkpoint[1], dict):
            self.critic.load_state_dict(checkpoint[1])
        else:
            self.critic = checkpoint[1].to(self.device)

        if isinstance(checkpoint[2], dict):
            self.target_critic.load_state_dict(checkpoint[2])
        else:
            self.target_critic = checkpoint[2].to(self.device)

        if checkpoint[3]:
            self._obs_rms = checkpoint[3]
            self._obs_rms = [x.to(self.device) for x in self._obs_rms]

        self.ret_rms = (
            checkpoint[4].to(self.device)
            if checkpoint[4] is not None
            else checkpoint[4]
        )
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

        if len(checkpoint) == 7:  # backwards compatible with older checkpoints
            self.actor_optimizer.load_state_dict(checkpoint[5])
            self.critic_optimizer.load_state_dict(checkpoint[6])

    def resume_from(self, path, cfg, epoch, step_count=None, loss=None):
        self.curr_epoch = epoch
        if loss:
            self.best_policy_loss = loss
        if step_count:
            self.step_count = step_count
        if self.multi_gpu:
            ep_tensor = torch.tensor(
                [self.curr_epoch, self.step_count, self.best_policy_loss],
                device=self.device,
            )
            dist.broadcast(ep_tensor, 0)
            if self.rank != 0:
                self.curr_epoch = int(ep_tensor[0].item())
                self.step_count = int(ep_tensor[1].item())
                self.best_policy_loss = ep_tensor[2].item()
        self.load(path, cfg)

    def close(self):
        self.writer.close()
