# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
import sys

import torch

from .dflex_env import DFlexEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dflex as df

import numpy as np

np.set_printoptions(precision=5, linewidth=256, suppress=True)

import dflex.envs.load_utils as lu
import dflex.envs.torch_utils as tu


class CheetahEnv(DFlexEnv):
    def __init__(
        self,
        render=False,
        device="cuda:0",
        num_envs=4096,
        seed=0,
        episode_length=1000,
        no_grad=True,
        stochastic_init=False,
        MM_caching_frequency=16,
        early_termination=True,
        jacobian=False,
        contact_ke=2.0e4,
        contact_kd=None,  #  1.0e3,
        logdir=None,
        nan_state_fix=False,
        jacobian_norm=None,
        reset_all=False,
    ):
        num_obs = 17
        num_act = 6

        super(CheetahEnv, self).__init__(
            num_envs,
            num_obs,
            num_act,
            episode_length,
            MM_caching_frequency,
            seed,
            no_grad,
            render,
            nan_state_fix,
            jacobian_norm,
            reset_all,
            stochastic_init,
            jacobian,
            device,
        )

        self.early_termination = early_termination
        self.contact_ke = contact_ke
        self.contact_kd = contact_kd if contact_kd is not None else contact_ke / 10.0

        self.init_sim()

        # other parameters
        self.action_strength = 200.0
        self.action_penalty = -1e-1

        # TODO logdir shouldn't need to be passed in here
        self.setup_visualizer(logdir)

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0 / 60.0
        self.sim_substeps = 16
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 9
        self.num_joint_qd = 9

        self.x_unit_tensor = tu.to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))

        self.start_rotation = torch.tensor(
            [0.0], device=self.device, requires_grad=False
        )

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()

        self.start_pos = []
        self.start_joint_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        start_height = -0.2

        asset_folder = os.path.join(os.path.dirname(__file__), "assets")
        for i in range(self.num_environments):
            link_start = len(self.builder.joint_type)

            lu.parse_mjcf(
                os.path.join(asset_folder, "half_cheetah.xml"),
                self.builder,
                density=1000.0,
                stiffness=0.0,
                damping=1.0,
                contact_ke=self.contact_ke,
                contact_kd=self.contact_kd,
                contact_kf=1e3,
                contact_mu=1.0,
                limit_ke=1e3,
                limit_kd=1e1,
                armature=0.1,
                radians=True,
                load_stiffness=True,
            )

            self.builder.joint_X_pj[link_start] = df.transform(
                (0.0, 1.0, 0.0),
                df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
            )

            # base transform
            self.start_pos.append([0.0, start_height])

            # set joint targets to rest pose in mjcf
            self.builder.joint_q[
                i * self.num_joint_q + 3 : i * self.num_joint_q + 9
            ] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.builder.joint_target[
                i * self.num_joint_q + 3 : i * self.num_joint_q + 9
            ] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.start_pos = tu.to_torch(self.start_pos, device=self.device)
        self.start_joint_q = tu.to_torch(self.start_joint_q, device=self.device)

        # finalize model
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor(
            (0.0, -9.81, 0.0), dtype=torch.float32, device=self.device
        )

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()

        if self.model.ground:
            self.model.collide(self.state)

    def unscale_act(self, action):
        return action * self.action_strength

    def set_act(self, action):
        self.state.joint_act.view(self.num_envs, -1)[:, 3:] = action

    def compute_termination(self, obs, act):
        # Cheetah has no early termination
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return termination

    def static_init_func(self, env_ids):
        xy = self.start_pos[env_ids]
        th = self.start_rotation.repeat(len(env_ids), 1)
        joints = self.start_joint_q.repeat(len(env_ids), 1)
        joint_q = torch.cat((xy, th, joints), dim=-1)
        joint_qd = torch.zeros((len(env_ids), self.num_joint_qd), device=self.device)
        return joint_q, joint_qd

    def stochastic_init_func(self, env_ids):
        """Method for computing stochastic init state"""
        xy = (
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:2]
            + 0.1 * (torch.rand(size=(len(env_ids), 2), device=self.device) - 0.5) * 2.0
        )
        z = (torch.rand((len(env_ids), 1), device=self.device) - 0.5) * 0.2

        joints = (
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:]
            + 0.1
            * (
                torch.rand(
                    size=(len(env_ids), self.num_joint_q - 3),
                    device=self.device,
                )
                - 0.5
            )
            * 2.0
        )
        joint_q = torch.cat((xy, z, joints), dim=-1)
        joint_qd = 0.5 * (
            torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5
        )
        return joint_q, joint_qd

    def set_state_act(self, obs, act):
        self.state.joint_q.view(self.num_envs, -1)[:, 1:] = obs[:, :8]
        self.state.joint_qd.view(self.num_envs, -1)[:, :] = obs[:, 8:]
        self.state.joint_act.view(self.num_envs, -1)[:, 3:] = act

    def observation_from_state(self, state):
        return torch.cat(
            [
                state.joint_q.view(self.num_envs, -1)[:, 1:],
                state.joint_qd.view(self.num_envs, -1),
            ],
            dim=-1,
        )

    def calculate_reward(self, obs, act):
        progress_reward = obs[:, 8]
        act_penalty = torch.sum(act**2, dim=-1) * self.action_penalty

        return progress_reward + act_penalty
