# Added option to resume training and loa Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import math
from hydra.utils import instantiate
import os
import sys
import time

import torch
from omegaconf import DictConfig, OmegaConf
from rl_games.common import schedulers
from rl_games.algos_torch import torch_ext
from shac.utils.average_meter import AverageMeter
from shac.utils.common import *
from shac.utils.dataset import QCriticDataset
from shac.utils.running_mean_std import RunningMeanStd
from shac.utils.time_report import TimeReport
from shac.models import actor as actor_models
from shac.models import critic as critic_model
import shac.utils.torch_utils as tu
from tensorboardX import SummaryWriter
import torch.distributed as dist
from torch.nn.utils.clip_grad import clip_grad_norm_


project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)


class SHAC:
    """SHAC algorithm implementation"""

    def __init__(self, cfg: DictConfig):
        seeding(cfg.general.seed)
        self.env = instantiate(cfg.env.config)

        print("num_envs = ", self.env.num_envs)
        print("num_actions = ", self.env.num_actions)
        print("num_obs = ", self.env.num_obs)

        self.multi_gpu = cfg.general.multi_gpu
        self.rank = 0
        self.rank_size = 1

        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)

            self.device_name = "cuda:" + str(self.rank)
            cfg.general.device = self.device_name
            if self.rank != 0:
                cfg.alg.params.config.print_stats = False
                # cfg["params"]["config"]["lr_schedule"] = None

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length
        self.device = cfg.general.device

        self.gamma = cfg.alg.params.config.gamma

        self.critic_method = cfg.alg.params.config.critic_method
        if self.critic_method == "td-lambda":
            self.lam = cfg.alg.params.config.lam

        self.steps_num = cfg.alg.params.config.steps_num
        self.max_epochs = cfg.alg.params.config.max_epochs
        self.actor_lr = float(cfg.env.shac2.actor_lr)
        self.critic_lr = float(cfg.env.shac2.critic_lr)
        self.lr_schedule = cfg.alg.params.config.lr_schedule

        self.is_adaptive_lr = self.lr_schedule == "adaptive"
        self.is_linear_lr = self.lr_schedule == "linear"

        if self.is_adaptive_lr:
            self.scheduler = instantiate(cfg.alg.params.default_adaptive_scheduler)
        elif self.is_linear_lr:
            self.scheduler = instantiate(cfg.alg.params.default_linear_scheduler)
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.target_critic_alpha = cfg.alg.params.config.target_critic_alpha

        self._obs_rms = None
        self.curr_epoch = 0
        self.sub_traj_per_epoch = math.ceil(self.max_episode_length / self.steps_num)
        # number of epochs of no improvement for early stopping
        self.early_stopping_patience = cfg.alg.params.config.early_stopping_patience
        if cfg.alg.params.config.obs_rms:
            # generate obs_rms for each subtrajectory
            self._obs_rms = [
                RunningMeanStd(shape=(self.num_obs), device=self.device)
                # for _ in range(self.sub_traj_per_epoch)
            ]

        self.ret_rms = None
        if cfg.alg.params.config.ret_rms:
            self.ret_rms = RunningMeanStd(shape=(), device=self.device)

        self.rew_scale = cfg.alg.params.config.rew_scale

        self.critic_iterations = cfg.alg.params.config.critic_iterations
        self.num_batch = cfg.alg.params.config.num_batch
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.name = cfg.alg.params.config.name

        self.truncate_grad = cfg.alg.params.config.truncate_grads
        self.grad_norm = cfg.alg.params.config.grad_norm

        if cfg.general.train:
            self.log_dir = cfg.general.logdir
            if not self.multi_gpu or self.rank == 0:
                os.makedirs(self.log_dir, exist_ok=True)
                with open(os.path.join(self.log_dir, "cfg.yaml"), "w") as f:
                    f.write(OmegaConf.to_yaml(cfg))
                self.writer = SummaryWriter(os.path.join(self.log_dir, "log"))
            # save interval
            self.save_interval = cfg.alg.params.config.save_interval
            # stochastic inference
            self.stochastic_evaluation = True
        else:
            self.stochastic_evaluation = not cfg.alg.params.config.player.determenistic
            self.steps_num = self.env.episode_length

        # create actor critic network
        self.actor = instantiate(
            cfg.alg.params.network.actor,
            obs_dim=self.num_obs,
            action_dim=self.num_actions,
            device=self.device,
        )
        self.critic = instantiate(
            cfg.alg.params.network.critic,
            obs_dim=self.num_obs,
            action_dim=self.num_actions,
            device=self.device,
        )
        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.target_critic = copy.deepcopy(self.critic)

        # initialize optimizer
        self.actor_optimizer = instantiate(cfg.alg.params.config.actor_optimizer, params=self.actor.parameters())
        self.critic_optimizer = instantiate(cfg.alg.params.config.critic_optimizer, params=self.critic.parameters())

        self.mixed_precision = cfg.general.mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # replay buffer
        self.obs_buf = torch.zeros(
            (self.steps_num, self.num_envs, self.num_obs),
            dtype=torch.float32,
            device=self.device,
        )
        self.act_buf = torch.zeros(
            (self.steps_num, self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.rew_buf = torch.zeros((self.steps_num, self.num_envs), dtype=torch.float32, device=self.device)
        self.done_mask = torch.zeros((self.steps_num, self.num_envs), dtype=torch.float32, device=self.device)
        self.next_values = torch.zeros((self.steps_num, self.num_envs), dtype=torch.float32, device=self.device)
        self.target_values = torch.zeros((self.steps_num, self.num_envs), dtype=torch.float32, device=self.device)
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

        if cfg.general.train:
            self.save("init_policy")

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # loss variables
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.best_policy_loss = np.inf
        self.best_policy_epoch = 0
        self.actor_loss = np.inf
        self.value_loss = np.inf

        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.score_keys = cfg.alg.params.config.score_keys
        self.episode_scores_meter_map = {
            key + "_final": AverageMeter(1, 100).to(self.device) for key in self.score_keys
        }

        # timer
        self.time_report = TimeReport()

    @property
    def obs_rms(self):
        if self._obs_rms is None:
            return self._obs_rms
        return self._obs_rms[0]  # self.curr_epoch % self.sub_traj_per_epoch]

    def compute_actor_loss(self, deterministic=False):
        rew_acc = torch.zeros((self.steps_num + 1, self.num_envs), dtype=torch.float32, device=self.device)
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        next_values = torch.zeros((self.steps_num + 1, self.num_envs), dtype=torch.float32, device=self.device)
        # next_values_model_free = torch.zeros(
        #     (self.steps_num + 1, self.num_envs), dtype=torch.float32, device=self.device
        # )

        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        # actor_model_free_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

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

        # copy previous state mus, sigma to old_mus, old_simgas
        if self.curr_epoch > 1:
            self.old_mus[:] = self.mus.clone()
            self.old_sigmas[:] = self.sigmas.clone()

        self.ep_kls = []
        next_actions = torch.tanh(self.actor(obs, deterministic=deterministic))

        for i in range(self.steps_num):
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[i] = obs.clone()

            # normalized sampled action: pi(s)
            actions = next_actions

            with torch.no_grad():
                self.act_buf[i] = actions.clone()

            with torch.no_grad():
                _, mus_i, sigmas_i = self.actor.forward_with_dist(obs, deterministic=True)
                self.mus[i, :], self.sigmas[i, :] = mus_i.clone(), sigmas_i.clone()

            obs, rew, done, extra_info = self.env.step(actions)

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
                    self.ret[:] = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)

                rew = rew / torch.sqrt(ret_var + 1e-6)

            self.episode_length += 1

            # done = done.clone() | extra_info.get("contact_changed", torch.zeros_like(done))
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            next_actions = torch.tanh(self.actor(obs, deterministic=True))
            next_values[i + 1] = self.target_critic(obs, next_actions).squeeze(-1)
            # next_values_model_free[i + 1] = self.target_critic(obs.detach(), next_actions.detach()).squeeze(-1)
            # next_values_model_free[i + 1] = torch.minimum(
            #     self.critic(obs.requires_grad_(False), actions).squeeze(-1),
            #     self.target_critic(obs.require_grad_(False), actions).squeeze(-1),
            # )

            # zero next_values for done envs with inf, nan, or >1e6 values in obs_before_reset
            # or early termination

            if done_env_ids.shape[0] > 0:
                zero_next_values = torch.where(
                    torch.isnan(extra_info["obs_before_reset"][done_env_ids]).any(dim=-1)
                    | torch.isinf(extra_info["obs_before_reset"][done_env_ids]).any(dim=-1)
                    | (torch.abs(extra_info["obs_before_reset"][done_env_ids]) > 1e6).any(dim=-1)
                    | (self.episode_length[done_env_ids] < self.max_episode_length),
                    torch.ones_like(done_env_ids, dtype=bool),
                    torch.zeros_like(done_env_ids, dtype=bool),
                )
                zero_next_values, assign_next_values = done_env_ids[zero_next_values], done_env_ids[~zero_next_values]
                if zero_next_values.shape[0] > 0:
                    next_values[i + 1, zero_next_values] = 0.0
                    # next_values_model_free[i + 1, zero_next_values] = 0.0
                # use terminal value critic to estimate the long-term performance
                if assign_next_values.shape[0] > 0:
                    if self.obs_rms is not None:
                        real_obs = obs_rms.normalize(extra_info["obs_before_reset"][assign_next_values])
                        real_act = actions[assign_next_values]
                    else:
                        real_obs = extra_info["obs_before_reset"][assign_next_values]
                        real_act = actions[assign_next_values]
                    next_values[i + 1, assign_next_values] = self.critic(real_obs, real_act).squeeze(-1)
                    # next_values_model_free[i + 1, assign_next_values] = self.critic(
                    #     real_obs.detach(), real_actions
                    # ).squeeze(-1)
            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print("next value error")
                if self.multi_gpu:
                    dist.destroy_process_group()
                raise ValueError

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            if i < self.steps_num - 1:
                actor_loss += (
                    -rew_acc[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]
                ).sum()
                # actor_model_free_loss += (
                #     -rew_acc[i + 1, done_env_ids].detach()
                #     - self.gamma * gamma[done_env_ids] * next_values_model_free[i + 1, done_env_ids]
                # ).sum()
            else:
                # terminate all envs at the end of optimization iteration
                actor_loss += (-rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]).sum()
                # actor_model_free_loss += (
                #     -rew_acc[i + 1, :].detach() - self.gamma * gamma * next_values_model_free[i + 1, :]
                # ).sum()

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
                    self.episode_discounted_loss_meter.update(self.episode_discounted_loss[done_env_ids])
                    self.episode_length_meter.update(self.episode_length[done_env_ids])
                    for k, v in filter(lambda x: x[0] in self.score_keys, extra_info.items()):
                        self.episode_scores_meter_map[k + "_final"].update(v[done_env_ids])
                    for done_env_id in done_env_ids:
                        if self.episode_loss[done_env_id] > 1e6 or self.episode_loss[done_env_id] < -1e6:
                            print("ep loss error")
                            if self.multi_gpu:
                                dist.destroy_process_group()
                            raise ValueError

                        self.episode_loss_his.append(self.episode_loss[done_env_id].item())
                        self.episode_discounted_loss_his.append(self.episode_discounted_loss[done_env_id].item())
                        self.episode_length_his.append(self.episode_length[done_env_id].item())
                        self.episode_loss[done_env_id] = 0.0
                        self.episode_discounted_loss[done_env_id] = 0.0
                        self.episode_length[done_env_id] = 0
                        self.episode_gamma[done_env_id] = 1.0

        # if first epoch, clone old_mus, sigmas
        if self.curr_epoch == 1:
            self.old_mus[:] = self.mus.clone()
            self.old_sigmas[:] = self.sigmas.clone()

        actor_loss /= self.steps_num * self.num_envs
        # actor_model_free_loss /= self.steps_num * self.num_envs

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)
            # actor_model_free_loss = actor_model_free_loss * torch.sqrt(ret_var + 1e-6)

        self.actor_loss = actor_loss.detach().cpu().item()
        # self.actor_model_free_loss = actor_loss.detach().cpu().item()

        self.step_count += self.steps_num * self.num_envs

        return actor_loss, None

    @torch.no_grad()
    def evaluate_policy(self, num_games, deterministic=False):
        episode_length_his = []
        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        episode_length = torch.zeros(self.num_envs, dtype=int, device=self.device)
        episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        episode_discounted_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        obs = self.env.reset()

        games_cnt = 0
        while games_cnt < num_games:
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)

            actions = torch.tanh(self.actor(obs, deterministic=deterministic))

            obs, rew, done, _ = self.env.step(actions)

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
                    episode_discounted_loss_his.append(episode_discounted_loss[done_env_id].item())
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
                    self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i]))
                    + self.rew_buf[i]
                )
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError

    def compute_critic_loss(self, batch_sample):
        predicted_values = self.critic(batch_sample["obs"], batch_sample["act"]).squeeze(-1)
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
        ) = self.evaluate_policy(num_games=num_games, deterministic=not self.stochastic_evaluation)
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

        self.episode_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        def actor_closure():
            self.actor_optimizer.zero_grad(set_to_none=True)

            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")

            # use autoscaling for mixed precision
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                actor_loss, _ = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")
            self.scaler.scale(actor_loss).backward()
            # actor_loss.backward()
            self.time_report.end_timer("backward simulation")

            with torch.no_grad():
                # unscale here to get grad norm before clipping
                self.scaler.unscale_(self.actor_optimizer)
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                self.clip_gradients(self.actor.parameters(), self.actor_optimizer, unscale=False)
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters())
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()

                # sanity check
                if torch.isnan(self.grad_norm_before_clip) or self.grad_norm_before_clip > 1000000.0:
                    print("shac training crashed due to unstable gradient")
                    # torch.save(env_state, os.path.join(self.log_dir, "bad_state.pt"))
                    print("NaN gradient")
                    if not self.multi_gpu or self.rank == 0:
                        self.save("crashed")
                    if self.multi_gpu:
                        dist.destroy_process_group()
                    raise ValueError

            self.time_report.end_timer("compute actor loss")

            return

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
                    actor_lr, _ = self.scheduler.update(actor_lr, None, self.curr_epoch, None, None)
                    critic_lr, _ = self.scheduler.update(critic_lr, None, self.curr_epoch, None, None)

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
            elif self.is_adaptive_lr and len(self.ep_kls) > 0:
                av_kls = torch_ext.mean_list(self.ep_kls)
                if self.multi_gpu:
                    dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                    av_kls /= self.rank_size

                self.actor_lr, self.entropy_coef = self.scheduler.update(
                    self.actor_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
                )
                self.critic_lr = self.actor_lr
                self.ep_kls = []
            else:
                lr = self.actor_lr

            # train actor
            self.act_buf = torch.zeros(
                (self.steps_num, self.num_envs, self.num_actions),
                dtype=torch.float32,
                device=self.device,
            )
            self.time_report.start_timer("actor training")
            actor_closure()
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = QCriticDataset(
                    self.batch_size,
                    self.obs_buf,
                    self.act_buf,
                    self.target_values,
                    drop_last=False,
                )
                # compute KL divergence of the current policy
                self.ep_kls.append(
                    torch_ext.policy_kl(
                        self.mus.detach(),
                        self.sigmas.detach(),
                        self.old_mus,
                        self.old_sigmas,
                    )
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

                    self.clip_gradients(self.critic.parameters(), self.critic_optimizer)
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.update()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1
                    # recompute Q-target
                    self.compute_target_values()

                self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
                print(
                    "value iter {}/{}, loss = {:7.6f}".format(j + 1, self.critic_iterations, self.value_loss),
                    end="\r",
                )
            del self.act_buf
            del dataset

            self.time_report.end_timer("critic training")

            self.iter_count += 1

            time_end_epoch = time.time()
            should_exit = False

            # update target critic
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
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
                mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()

                if mean_policy_loss < self.best_policy_loss:
                    print_info("save best policy with loss {:.2f}".format(mean_policy_loss))
                    self.save()
                    self.best_policy_loss = mean_policy_loss
                    self.best_policy_epoch = self.curr_epoch
                # number of episodes with no improvement
                else:
                    last_improved_ep = self.best_policy_epoch - self.curr_epoch
                    if last_improved_ep > self.early_stopping_patience:
                        should_exit = True

                self.writer.add_scalar("policy_loss/step", mean_policy_loss, self.step_count)
                self.writer.add_scalar("policy_loss/time", mean_policy_loss, time_elapse)
                self.writer.add_scalar("policy_loss/iter", mean_policy_loss, self.iter_count)
                self.writer.add_scalar("rewards/step", -mean_policy_loss, self.step_count)
                self.writer.add_scalar("rewards/time", -mean_policy_loss, time_elapse)
                self.writer.add_scalar("rewards/iter", -mean_policy_loss, self.iter_count)
                if self.score_keys and len(self.episode_scores_meter_map[self.score_keys[0] + "_final"]) > 0:
                    for score_key in self.score_keys:
                        score = self.episode_scores_meter_map[score_key + "_final"].get_mean()
                        self.writer.add_scalar("scores/{}/iter".format(score_key), score, self.iter_count)
                        self.writer.add_scalar("scores/{}/step".format(score_key), score, self.step_count)
                        self.writer.add_scalar("scores/{}/time".format(score_key), score, time_elapse)
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

                self.writer.add_scalar("best_policy_loss/step", self.best_policy_loss, self.step_count)
                self.writer.add_scalar("best_policy_loss/iter", self.best_policy_loss, self.iter_count)
                self.writer.add_scalar("episode_lengths/iter", mean_episode_length, self.iter_count)
                self.writer.add_scalar("episode_lengths/step", mean_episode_length, self.step_count)
                self.writer.add_scalar("episode_lengths/time", mean_episode_length, time_elapse)
                ac_stddev = self.actor.get_logstd().exp().mean().detach().cpu().item()
                self.writer.add_scalar("ac_std/iter", ac_stddev, self.iter_count)
                self.writer.add_scalar("ac_std/step", ac_stddev, self.step_count)
                self.writer.add_scalar("ac_std/time", ac_stddev, time_elapse)
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
                    self.steps_num * self.num_envs * self.rank_size / (time_end_epoch - time_start_epoch),
                    self.value_loss,
                    self.grad_norm_before_clip,
                    self.grad_norm_after_clip,
                )
            )
            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(self.name + "policy_iter{}_reward{:.3f}".format(self.iter_count, -mean_policy_loss))

            if should_exit:
                break

        self.time_report.end_timer("algorithm")

        self.time_report.report()

        if self.rank == 0 or not self.multi_gpu:
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
            self.run(self.num_envs)
            self.close()

        if self.multi_gpu and should_exit:
            dist.destroy_process_group()

    def clip_gradients(self, parameters, optimizer, unscale=True):
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
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.rank_size
                    )
                    offset += param.numel()

        if self.truncate_grad:
            if unscale:
                self.scaler.unscale_(optimizer)
            clip_grad_norm_(parameters, self.grad_norm)

    def play(self, cfg):
        self.load(cfg.alg.params.general.checkpoint, cfg)
        self.run(cfg.alg.params.config.player.games_num)

    def save(self, filename=None):
        if filename is None:
            filename = "best_policy"
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "obs_rms": self._obs_rms,
                "ret_rms": self.ret_rms,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "mus": self.mus,
                "sigmas": self.sigmas,
                "old_mus": self.old_mus,
                "old_sigmas": self.old_sigmas,
            },
            os.path.join(self.log_dir, "{}.pt".format(filename)),
        )

    def load(self, path, cfg, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

        self.target_critic.load_state_dict(checkpoint["target_critic"])

        if checkpoint["obs_rms"] is not None:
            self._obs_rms = checkpoint["obs_rms"]
            self._obs_rms = [x.to(self.device) for x in self._obs_rms]
        else:
            self._obs_rms = None

        self.ret_rms = checkpoint["ret_rms"].to(self.device) if checkpoint["ret_rms"] is not None else None
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.mus = checkpoint["mus"].to(self.device)
        self.sigmas = checkpoint["sigmas"].to(self.device)
        self.old_mus = checkpoint["old_mus"].to(self.device)
        self.old_sigmas = checkpoint["old_sigmas"].to(self.device)

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
