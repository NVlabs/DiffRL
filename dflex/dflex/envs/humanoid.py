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

class HumanoidEnv(DFlexEnv):
    def __init__(
        self,
        render=False,
        device="cuda:0",
        num_envs=4096,
        episode_length=1000,
        no_grad=True,
        stochastic_init=False,
        MM_caching_frequency=1,
        early_termination=True,
        jacobian=False,
        contact_ke=2.0e4,
        contact_kd=None,
        logdir=None,
        nan_state_fix=True,  # humanoid env needs this
        jacobian_norm=None,
        termination_height=0.74,
        action_penalty=-0.002,
        joint_vel_obs_scaling=0.1,
        termination_tolerance=0.1,
        height_rew_scale=10.0,
        up_rew_scale=0.1,
        heading_rew_scale=1.0,
        height_rew_type="xu"
    ):
        num_obs = 76
        num_act = 21

        super(HumanoidEnv, self).__init__(
            num_envs,
            num_obs,
            num_act,
            episode_length,
            MM_caching_frequency,
            no_grad,
            render,
            nan_state_fix,
            jacobian_norm,
            stochastic_init,
            jacobian,
            device,
        )

        self.early_termination = early_termination
        self.contact_ke = contact_ke
        self.contact_kd = contact_kd if contact_kd is not None else contact_ke / 10.0

        self.init_sim()

        # other parameters
        self.termination_height = termination_height
        self.motor_strengths = [
            200,
            200,
            200,
            200,
            200,
            600,
            400,
            100,
            100,
            200,
            200,
            600,
            400,
            100,
            100,
            100,
            100,
            200,
            100,
            100,
            200,
        ]

        self.motor_scale = 0.35

        self.motor_strengths = tu.to_torch(
            self.motor_strengths,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        ).repeat((self.num_envs, 1))

        self.action_penalty = action_penalty
        self.joint_vel_obs_scaling = joint_vel_obs_scaling
        self.termination_tolerance = termination_tolerance
        self.height_rew_scale = height_rew_scale
        self.up_rew_scale = up_rew_scale
        self.heading_rew_scale = heading_rew_scale
        self.height_rew_type = height_rew_type

        self.setup_visualizer(logdir)

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0 / 60.0
        self.sim_substeps = 48
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 28
        self.num_joint_qd = 27

        self.x_unit_tensor = tu.to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))

        self.start_rot = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5)
        self.start_rotation = tu.to_torch(
            self.start_rot, device=self.device, requires_grad=False
        )

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat(
            (self.num_envs, 1)
        )

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = tu.to_torch(
            [200.0, 0.0, 0.0], device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))

        self.start_pos = []

        if self.visualize:
            self.env_dist = 2.5
        else:
            self.env_dist = 0.0  # set to zero for training for numerical consistency

        start_height = 1.35

        asset_folder = os.path.join(os.path.dirname(__file__), "assets")
        for i in range(self.num_environments):
            lu.parse_mjcf(
                os.path.join(asset_folder, "humanoid.xml"),
                self.builder,
                stiffness=5.0,
                damping=0.1,
                contact_ke=self.contact_ke,
                contact_kd=self.contact_kd,
                contact_kf=1.0e3,
                contact_mu=0.75,
                limit_ke=1.0e3,
                limit_kd=1.0e1,
                armature=0.007,
                load_stiffness=True,
                load_armature=True,
            )

            # base transform
            start_pos_z = i * self.env_dist
            self.start_pos.append([0.0, start_height, start_pos_z])

            self.builder.joint_q[
                i * self.num_joint_q : i * self.num_joint_q + 3
            ] = self.start_pos[-1]
            self.builder.joint_q[
                i * self.num_joint_q + 3 : i * self.num_joint_q + 7
            ] = self.start_rot

        num_q = int(len(self.builder.joint_q) / self.num_environments)
        num_qd = int(len(self.builder.joint_qd) / self.num_environments)
        print(num_q, num_qd)

        print("Start joint_q: ", self.builder.joint_q[0:num_q])

        self.start_joint_q = self.builder.joint_q[7:num_q].copy()

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

        num_act = int(len(self.state.joint_act) / self.num_environments) - 6
        print("num_act = ", num_act)

        if self.model.ground:
            self.model.collide(self.state)

    def unscale_act(self, action):
        return action * self.motor_scale * self.motor_strengths

    def set_act(self, action):
        self.state.joint_act.view(self.num_envs, -1)[:, 6:] = action

    def compute_termination(self, obs, act):
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # an ugly fix for simulation nan values
        joint_q = self.state.joint_q.view(self.num_environments, -1)
        joint_qd = self.state.joint_qd.view(self.num_environments, -1)

        nonfinite_mask = ~(torch.isfinite(obs).sum(-1) > 0)
        nonfinite_mask = nonfinite_mask | ~(torch.isfinite(joint_q).sum(-1) > 0)
        nonfinite_mask = nonfinite_mask | ~(torch.isfinite(joint_qd).sum(-1) > 0)

        invalid_value_mask = (torch.abs(joint_q) > 1e6).sum(-1) > 0
        invalid_value_mask = (
            invalid_value_mask | (torch.abs(joint_qd) > 1e6).sum(-1) > 0
        )

        termination = termination | nonfinite_mask | invalid_value_mask

        if self.early_termination:
            termination = termination | (obs[:, 0] < self.termination_height)
        return termination

    def static_init_func(self, env_ids):
        xyz = self.start_pos[env_ids]
        quat = self.start_rotation.repeat(len(env_ids), 1)
        joints = self.start_joint_q.repeat(len(env_ids), 1)
        joint_q = torch.cat((xyz, quat, joints), dim=-1)
        joint_qd = torch.zeros((len(env_ids), self.num_joint_qd), device=self.device)
        return joint_q, joint_qd

    def stochastic_init_func(self, env_ids):
        """Method for computing stochastic init state"""
        xyz = (
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3]
            + 0.1 * (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.0
        )
        angle = (torch.rand(len(env_ids), device=self.device) - 0.5) * np.pi / 12.0
        axis = torch.nn.functional.normalize(
            torch.rand((len(env_ids), 3), device=self.device) - 0.5
        )
        quat = tu.quat_mul(
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7],
            tu.quat_from_angle_axis(angle, axis),
        )

        joints = (
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:]
            + 0.2
            * (
                torch.rand(
                    size=(len(env_ids), self.num_joint_q - 7),
                    device=self.device,
                )
                - 0.5
            )
            * 2.0
        )
        joint_q = torch.cat((xyz, quat, joints), dim=-1)
        joint_qd = 0.5 * (
            torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5
        )
        return joint_q, joint_qd

    def set_state_act(self, obs, act):
        self.state.joint_q.view(self.num_envs, -1)[:, 1:2] = obs[:, [0]]
        self.state.joint_q.view(self.num_envs, -1)[:, 3:7] = obs[:, 1:5]
        self.state.joint_qd.view(self.num_envs, -1)[:, 3:6] = obs[:, 5:8]
        self.state.joint_q.view(self.num_envs, -1)[:, 7:] = obs[:, 11:32]
        self.state.joint_qd.view(self.num_envs, -1)[:, 6:] = (
            obs[:, 32:53] / self.joint_vel_obs_scaling
        )
        self.state.joint_act.view(self.num_envs, -1)[:, 6:] = act

    def observation_from_state(self, state):
        torso_pos = state.joint_q.view(self.num_envs, -1)[:, 0:3]
        torso_rot = state.joint_q.view(self.num_envs, -1)[:, 3:7]
        lin_vel = state.joint_qd.view(self.num_envs, -1)[:, 3:6]
        ang_vel = state.joint_qd.view(self.num_envs, -1)[:, 0:3]

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(torso_pos, ang_vel, dim=-1)

        to_target = self.targets + self.start_pos - torso_pos
        to_target[:, 1] = 0.0

        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot)

        up_vec = tu.quat_rotate(torso_quat, self.basis_vec1)
        heading_vec = tu.quat_rotate(torso_quat, self.basis_vec0)

        actions = state.joint_act.view(self.num_envs, -1)[:, 6:].clone()
        actions = actions / self.motor_scale / self.motor_strengths

        return torch.cat(
            [
                torso_pos[:, 1:2],  # 0
                torso_rot,  # 1:5
                lin_vel,  # 5:8
                ang_vel,  # 8:11
                state.joint_q.view(self.num_envs, -1)[:, 7:],  # 11:32
                self.joint_vel_obs_scaling
                * state.joint_qd.view(self.num_envs, -1)[:, 6:],  # 32:53
                up_vec[:, 1:2],  # 53:54
                (heading_vec * target_dirs).sum(dim=-1).unsqueeze(-1),  # 54:55
                actions,  # 55:76
            ],
            dim=-1,
        )
    
    def xu_func(self, x, term, tol, pos_scale, neg_scale):
        x = x - term - tol
        x = torch.clip(x, -1.0, tol)
        x = torch.where(x < 0.0, neg_scale * x * x, x)
        x = torch.where(x >= 0.0, pos_scale * x, x)
        return x
    
    def calculate_reward(self, obs, act):
        up_reward = self.up_rew_scale * obs[:, 53]
        heading_reward = self.heading_rew_scale * obs[:, 54]

        if self.height_rew_type == "xu":
            height_reward = self.xu_func(
                obs[:, 0],
                self.termination_height,
                self.termination_tolerance,
                self.height_rew_scale,
                -200.0,
            )
        elif self.height_rew_type == "linear":
            error = obs[:, 0] - self.termination_height
            height_reward = self.heading_rew_scale * error
        elif self.height_rew_type == "exponent":
            error = obs[:, 0] - self.termination_height
            height_reward = self.height_rew_scale * (-torch.exp(-error))
        elif self.height_rew_type == "log-barrier":
            error = obs[:, 0] - self.termination_height
            error = torch.clip(error, -1.0,)
            height_reward = self.height_rew_scale * torch.log(error+1)
        elif self.height_rew_type == "log-barrier-clip":
            error = obs[:, 0] - self.termination_height
            error = torch.clip(error, -1.0, self.termination_tolerance)
            height_reward = self.height_rew_scale * torch.log(error+1)
        else:
            raise NotImplemented

        progress_reward = obs[:, 5]
        self.primal = progress_reward.detach()
        act_penalty = torch.sum(act**2, dim=-1) * self.action_penalty

        return (
            progress_reward + up_reward + heading_reward + height_reward + act_penalty
        )
