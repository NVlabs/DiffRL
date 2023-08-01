# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
from abc import ABC, abstractmethod

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from shac.utils import torch_utils as tu

import dflex as df
import xml.etree.ElementTree as ET

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

from gym import spaces


class DFlexEnv:
    def __init__(
        self,
        num_envs,
        num_obs,
        num_act,
        episode_length,
        MM_caching_frequency=1,
        seed=0,
        no_grad=True,
        render=False,
        device="cuda:0",
    ):
        self.seed = seed

        self.no_grad = no_grad
        df.config.no_grad = self.no_grad

        self.episode_length = episode_length

        self.device = device

        self.visualize = render

        self.sim_time = 0.0

        self.num_frames = 0  # record the number of frames for rendering

        self.num_environments = num_envs
        self.num_agents = 1

        self.MM_caching_frequency = MM_caching_frequency

        # initialize observation and action space
        self.num_observations = num_obs
        self.num_actions = num_act

        self.obs_space = spaces.Box(
            np.ones(self.num_observations) * -np.Inf,
            np.ones(self.num_observations) * np.Inf,
        )
        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0
        )

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations),
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )

        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )

    @abstractmethod
    def observation_from_state(self, state):
        pass

    @abstractmethod
    def set_act(self, act):
        pass

    @abstractmethod
    def set_state_act(self, obs, act):
        pass

    @abstractmethod
    def unscale_act(self, action):
        pass

    @abstractmethod
    def compute_termination(self, obs, act):
        pass

    @abstractmethod
    def calculate_reward(self, obs, act):
        pass

    @abstractmethod
    def static_init_func(self, env_ids):
        pass

    @abstractmethod
    def stochastic_init_func(self, env_ids):
        pass

    def step(self, actions, play=False):
        actions = actions.view((self.num_envs, self.num_actions))
        actions = torch.clip(actions, -1.0, 1.0)
        unscaled_actions = self.unscale_act(actions)
        self.set_act(unscaled_actions)

        # NOTE: Potentially needed for humanoid env
        ##### an ugly fix for simulation nan values #### # reference: https://github.com/pytorch/pytorch/issues/15131
        # def create_hook():
        #     def hook(grad):
        #         torch.nan_to_num(grad, 0.0, 0.0, 0.0, out=grad)

        #     return hook

        # if self.state.joint_q.requires_grad:
        #     self.state.joint_q.register_hook(create_hook())
        # if self.state.joint_qd.requires_grad:
        #     self.state.joint_qd.register_hook(create_hook())
        # if actions.requires_grad:
        #     actions.register_hook(create_hook())
        #################################################

        next_state = self.integrator.forward(
            self.model,
            self.state,
            self.sim_dt,
            self.sim_substeps,
            self.MM_caching_frequency,
        )

        # compute dynamics jacobians if requested
        if self.jacobian and not play:
            inputs = torch.cat((self.obs_buf.clone(), unscaled_actions.clone()), dim=1)
            inputs.requires_grad_(True)
            last_obs = inputs[:, : self.num_obs]
            act = inputs[:, self.num_obs :]
            self.set_state_act(last_obs, act)
            output = self.integrator.forward(
                self.model,
                self.state,
                self.sim_dt,
                self.sim_substeps,
                self.MM_caching_frequency,
                False,
            )
            outputs = self.observation_from_state(output)
            # TODO for some reason can only compute up to 11th dim
            jac = tu.jacobian2(outputs, inputs, max_out_dim=11)
            self.jacobians.append(jac.cpu().numpy())

        self.state = next_state
        self.sim_time += self.sim_dt

        self.progress_buf += 1
        self.num_frames += 1

        rew = self.calculate_reward(self.obs_buf, actions)
        self.obs_buf = self.observation_from_state(self.state)

        # Reset environments if exseeded horizon
        # NOTE: this is truncation
        truncation = self.progress_buf > self.episode_length - 1

        # Reset environments if agent has ended in a bad state based on heuristics
        # NOTE: this is termination
        termination = self.compute_termination(self.obs_buf, actions)

        extras = None
        if self.no_grad == False:
            extras = {
                "obs_before_reset": self.obs_buf.clone(),
                "episode_end": termination,
                "contact_count": self.state.contact_count.clone().detach(),
                "contact_forces": self.state.contact_f.clone().detach(),
            }

            if self.jacobian and not play:
                extras.update({"jacobian": jac.cpu().numpy()})
                # filename = f"/home/ignat/{self.__class__.__name__}_jacobians.npy"
                # np.save(filename, self.jacobians)

        # reset all environments which have been terminated
        done = termination | truncation
        env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.render()

        return self.obs_buf, rew, done, extras

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(
                    self.num_envs, dtype=torch.long, device=self.device
                )

        if env_ids is not None:
            # clone the state to avoid gradient error
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()

            # fixed start state
            joint_q, joint_qd = self.static_init_func(env_ids)
            self.state.joint_q.view(self.num_envs, -1)[env_ids] = joint_q
            self.state.joint_qd.view(self.num_envs, -1)[env_ids] = joint_qd

            # randomization
            if self.stochastic_init:
                joint_q, joint_qd = self.stochastic_init_func(env_ids)
                self.state.joint_q.view(self.num_envs, -1)[env_ids] = joint_q
                self.state.joint_qd.view(self.num_envs, -1)[env_ids] = joint_qd

            # clear action
            self.state.joint_act.view(self.num_envs, -1)[env_ids, :] = 0.0

            self.progress_buf[env_ids] = 0

            self.obs_buf = self.observation_from_state(self.state)

        return self.obs_buf

    def setup_visualizer(self, logdir=None):
        if self.visualize:
            filename = f"{logdir}/{self.__class__.__name__}_{self.num_envs}.usd"
            self.stage = Usd.Stage.CreateNew(filename)
            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

    def render(self, mode="human"):
        if self.visualize:
            self.render_time += self.dt
            self.renderer.update(self.state, self.render_time)

            render_interval = 1
            if self.num_frames == render_interval:
                try:
                    self.stage.Save()
                except:
                    print("USD save error")

                self.num_frames -= render_interval

    def clear_grad(self, checkpoint=None):
        """cut off the gradient from the current state to previous states"""

        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}
                checkpoint["joint_q"] = self.state.joint_q.clone()
                checkpoint["joint_qd"] = self.state.joint_qd.clone()
                checkpoint["joint_act"] = self.state.joint_act.clone()
                checkpoint["progress_buf"] = self.progress_buf.clone()

            self.state = self.model.state()
            self.state.joint_q = checkpoint["joint_q"]
            self.state.joint_qd = checkpoint["joint_qd"]
            self.state.joint_act = checkpoint["joint_act"]
            self.progress_buf = checkpoint["progress_buf"]

    def initialize_trajectory(self):
        """
        This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
        It has to be called every time the algorithm starts an episode and it returns the observation vectors
        """
        self.clear_grad()
        self.obs_buf = self.observation_from_state(self.state)
        return self.obs_buf

    def get_checkpoint(self):
        checkpoint = {}
        checkpoint["joint_q"] = self.state.joint_q.clone()
        checkpoint["joint_qd"] = self.state.joint_qd.clone()
        checkpoint["joint_act"] = self.state.joint_act.clone()
        checkpoint["progress_buf"] = self.progress_buf.clone()

        return checkpoint

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

    def get_state(self):
        return self.state.joint_q.clone(), self.state.joint_qd.clone()

    def reset_with_state(
        self, init_joint_q, init_joint_qd, env_ids=None, force_reset=True
    ):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(
                    self.num_envs, dtype=torch.long, device=self.device
                )

        if env_ids is not None:
            # fixed start state
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, :] = init_joint_q.view(
                -1, self.num_joint_q
            )[env_ids, :].clone()
            self.state.joint_qd.view(self.num_envs, -1)[
                env_ids, :
            ] = init_joint_qd.view(-1, self.num_joint_qd)[env_ids, :].clone()

            self.progress_buf[env_ids] = 0

            self.obs_buf = self.observation_from_state(self.state)

        return self.obs_buf
