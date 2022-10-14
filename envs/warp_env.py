import os
from gym import spaces
import torch
import numpy as np
import warp as wp
import warp.torch
import warp.sim
import warp.sim.render


class WarpEnv:
    def __init__(
        self,
        num_envs,
        num_obs,
        num_act,
        episode_length,
        seed=0,
        no_grad=True,
        render=True,
        device="cuda",
        env_name="warp_env",
    ):
        self.seed = seed
        self.no_grad = no_grad
        self.device = str(device).split(":")[0]
        self.visualize = render
        self.sim_time = 0.0
        self.num_frames = 0

        self.num_environments = num_envs
        self.env_name = env_name

        # initialize observation and action space
        self.num_observations = num_obs
        self.num_actions = num_act
        self.episode_length = episode_length

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
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float, requires_grad=False
        )
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        # end of the episode
        self.termination_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions),
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )

        self.extras = {}

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

    def get_state(self, return_act=False):
        """Returns model joint state (position and velocity)"""
        joint_q, joint_qd = (
            wp.to_torch(self.model.joint_q),
            wp.to_torch(self.model.joint_qd),
        )
        if not return_act:
            return joint_q, joint_qd
        joint_act = wp.to_torch(self.model.joint_act)
        return joint_q, joint_qd, joint_act

