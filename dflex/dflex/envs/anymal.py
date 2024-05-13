# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
import sys
import time
import urchin

import torch

from .dflex_env import DFlexEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dflex as df

import numpy as np

np.set_printoptions(precision=5, linewidth=256, suppress=True)

import dflex.envs.load_utils as lu
import dflex.envs.torch_utils as tu


class AnymalEnv(DFlexEnv):
    """
    TODO
    """

    def __init__(
        self,
        render=False,
        device="cuda:0",
        num_envs=64,
        episode_length=1000,
        no_grad=True,
        stochastic_init=False,
        MM_caching_frequency=16,
        early_termination=True,
        jacobian=False,
        logdir=None,
        nan_state_fix=False,
        jacobian_norm=None,
        termination_height=0.25,
        action_penalty=-0.005,
        up_rew_scale=0.1,
        heading_rew_scale=1.0,
        heigh_rew_scale=1.0,
    ):
        num_obs = 49
        num_act = 12

        super(AnymalEnv, self).__init__(
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
        self.init_sim()

        # MDP parameters
        self.action_strength = 200.0
        self.joint_vel_obs_scaling = 0.1

        # reward parameters
        self.termination_height = termination_height
        self.action_penalty = action_penalty
        self.up_rew_scale = up_rew_scale
        self.heading_rew_scale = heading_rew_scale
        self.heigh_rew_scale = heigh_rew_scale

        self.setup_visualizer(logdir)

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0 / 60.0
        self.sim_substeps = 16
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 12 + 7  # joints + root pose
        self.num_joint_qd = 12 + 6  # joints + root velocity

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
            [10000.0, 0.0, 0.0], device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))

        self.start_pos = []
        self.start_joint_q = [
            0.03,  # LF_HAA
            0.4,  # LF_HFE
            -0.8,  # LF_KFE
            -0.03,  # RF_HAA
            0.4,  # RF_HFE
            -0.8,  # RF_KFE
            0.03,  # LH_HAA
            -0.4,  # LH_HFE
            0.8,  # LH_KFE
            -0.03,  # RH_HAA
            -0.4,  # RH_HFE
            0.8,  # RH_KFE
        ]
        # Taken from IsaacGym
        # https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/Anymal.yaml

        if self.visualize:
            self.env_dist = 2.5
        else:
            self.env_dist = 0.0  # set to zero for training for numerical consistency

        start_height = 0.55

        asset_folder = os.path.join(os.path.dirname(__file__), "assets")
        filename = "anymal_c/urdf/anymal_minimal.urdf"
        now = time.time()
        # load URDF here for faster env initialisation when running many paralle envs
        robot = urchin.URDF.load(os.path.join(asset_folder, filename), lazy_load_meshes=True)
        for i in range(self.num_environments):
            start_pos = (0.0, start_height, 0.0 + self.env_dist * i)
            lu.urdf_load(
                self.builder,
                os.path.join(asset_folder, filename),
                df.transform(
                    start_pos,
                    self.start_rot,
                ),
                robot=robot,
                floating=True,
                stiffness=85.0,  # from config
                damping=2.0,  # from config
                shape_ke=2.0e3,
                shape_kd=5.0e2,
                shape_kf=1.0e2,
                shape_mu=0.75,
                limit_ke=1.0e3,
                limit_kd=1.0e1,
                armature=0.006,
            )
            self.start_pos.append(start_pos)

        total = time.time() - now
        print(f"Loading took {total:.2f}s")

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
        raise NotImplemented
        # torso position
        self.state.joint_q.view(self.num_envs, -1)[:, 1] = obs[:, 0]
        # torso rotation
        self.state.joint_q.view(self.num_envs, -1)[:, 3:7] = obs[:, 1:5]
        # linear velocity
        self.state.joint_qd.view(self.num_envs, -1)[:, 3:6] = obs[:, 5:8]
        # angular velocity
        self.state.joint_qd.view(self.num_envs, -1)[:, 0:3] = obs[:, 8:11]
        self.state.joint_q.view(self.num_envs, -1)[:, 7:] = obs[:, 11:19]
        self.state.joint_qd.view(self.num_envs, -1)[:, 6:] = obs[:, 19:27]
        self.state.joint_act.view(self.num_envs, -1)[:, 6:] = act

    def observation_from_state(self, state):
        # torse related things
        torso_pos = state.joint_q.view(self.num_envs, -1)[:, 0:3].clone()
        torso_rot = state.joint_q.view(self.num_envs, -1)[:, 3:7].clone()
        lin_vel = state.joint_qd.view(self.num_envs, -1)[:, 3:6].clone()
        ang_vel = state.joint_qd.view(self.num_envs, -1)[:, 0:3].clone()

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(torso_pos, ang_vel, dim=-1)

        to_target = self.targets + self.start_pos - torso_pos
        to_target[:, 1] = 0.0

        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot)

        up_vec = tu.quat_rotate(torso_quat, self.basis_vec1)
        heading_vec = tu.quat_rotate(torso_quat, self.basis_vec0)

        action = state.joint_act.view(self.num_envs, -1) / self.action_strength

        return torch.cat(
            [
                torso_pos[:, 1:2],  # 0 torso height
                torso_rot,  # 1:5
                lin_vel,  # 5:8
                ang_vel,  # 8:11
                # joint positions
                state.joint_q.view(self.num_envs, -1)[:, 7:],  # 11:23
                # joint velocities
                self.joint_vel_obs_scaling
                * state.joint_qd.view(self.num_envs, -1)[:, 6:],  # 23:35
                up_vec[:, 1:2],  # 35
                (heading_vec * target_dirs).sum(dim=-1).unsqueeze(-1),  # 36
                action[:, 6:].clone(),  # 36:48 - 12 actions
            ],
            dim=-1,
        )

    def calculate_reward(self, obs, act):
        up_rew = self.up_rew_scale * obs[:, 35]
        heading_rew = self.heading_rew_scale * obs[:, 36]
        height_rew = self.heigh_rew_scale * (obs[:, 0] - self.termination_height)
        progress_rew = obs[:, 5]  # forward velocity
        self.primal = progress_rew.detach()
        act_penalty = self.action_penalty * torch.sum(act**2, dim=-1)

        return progress_rew + up_rew + heading_rew + height_rew + act_penalty
