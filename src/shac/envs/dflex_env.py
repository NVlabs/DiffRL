# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

import dflex as df
import xml.etree.ElementTree as ET

from gym import spaces


class DFlexEnv:
    
    def __init__(self, num_envs, num_obs, num_act, episode_length, MM_caching_frequency = 1, seed=0, no_grad=True, render=False, device='cuda:0'):
        self.seed = seed

        self.no_grad = no_grad
        df.config.no_grad = self.no_grad

        self.episode_length = episode_length

        self.device = device

        self.visualize = render

        self.sim_time = 0.0

        self.num_frames = 0 # record the number of frames for rendering

        self.num_environments = num_envs
        self.num_agents = 1

        self.MM_caching_frequency = MM_caching_frequency
        
        # initialize observation and action space
        self.num_observations = num_obs
        self.num_actions = num_act

        self.obs_space = spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations), device=self.device, dtype=torch.float, requires_grad=False)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        # end of the episode
        self.termination_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device = self.device, dtype = torch.float, requires_grad = False)

        self.extras = {}

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

    def reset_with_state(self, init_joint_q, init_joint_qd, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # fixed start state
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, :] = init_joint_q.view(-1, self.num_joint_q)[env_ids, :].clone()
            self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = init_joint_qd.view(-1, self.num_joint_qd)[env_ids, :].clone()
            
            self.progress_buf[env_ids] = 0

            self.calculateObservations()

        return self.obs_buf