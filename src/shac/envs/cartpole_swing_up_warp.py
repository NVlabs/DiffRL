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

from .warp_env import WarpEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warp as wp

import numpy as np

np.set_printoptions(precision=5, linewidth=256, suppress=True)

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

from utils import torch_utils as tu
from utils.warp_utils import IntegratorSimulate


class CartPoleSwingUpWarpEnv(WarpEnv):
    def __init__(
        self,
        render=False,
        device="cuda",
        num_envs=1024,
        seed=0,
        episode_length=240,
        no_grad=True,
        stochastic_init=False,
        early_termination=False,
    ):

        num_obs = 5
        num_act = 1

        super(CartPoleSwingUpWarpEnv, self).__init__(
            num_envs, num_obs, num_act, episode_length, seed, no_grad, render, stochastic_init, device
        )

        self.early_termination = early_termination

        self.init_sim()

        # action parameters
        self.action_strength = 1000.0

        # loss related
        self.pole_angle_penalty = 1.0
        self.pole_velocity_penalty = 0.1

        self.cart_position_penalty = 0.05
        self.cart_velocity_penalty = 0.1

        self.cart_action_penalty = 0.

        # -----------------------
        # set up Usd renderer
        if self.visualize:
            stage_path = f"outputs/CartPoleSwingUpWarp_{self.num_envs}.usd"
            print(f"created stage, {stage_path}")
            self.stage = wp.sim.render.SimRenderer(self.model, stage_path)
            self.stage.draw_points = True
            self.stage.draw_springs = True
            self.stage.draw_shapes = True
            self.render_time = 0.0

    def init_sim(self):
        wp.init()
        self.dt = 1.0 / 60.0
        self.sim_substeps = 4
        self.sim_dt = self.dt

        if self.visualize:
            self.env_dist = 1.0
        else:
            self.env_dist = 0.0

        self.num_joint_q = 2
        self.num_joint_qd = 2

        asset_folder = os.path.join(os.path.dirname(__file__), "assets")
        self.articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/cartpole.urdf"),
            self.articulation_builder,
            xform=wp.transform(
                np.array((0.0, 0.0, 0.0)),
                wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
            ),
            floating=False,
            density=0,
            # armature=0.1,
            # stiffness=0.0,
            # damping=0.0,
            # shape_ke=1.0e4,
            shape_kd=1.0e4,
            # shape_kf=1.0e4,
            # shape_mu=1.0,
            # limit_ke=100,
            # limit_kd=1.0,
        )

        self.builder = wp.sim.ModelBuilder()

        for i in range(self.num_envs):
            self.builder.add_rigid_articulation(
                self.articulation_builder,
                xform=wp.transform(
                    np.array((0.0, 2.5, self.env_dist * i)),
                    wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
                ),
            )
            self.builder.joint_q[i * self.num_joint_q + 1] = -math.pi
            self.builder.joint_target[i * self.num_joint_q:(i+1) * self.num_joint_q] = [0., 0.]

        self.model = self.builder.finalize(str(self.device))
        self.model.ground = False

        self.model.joint_attach_ke = 10000.0
        self.model.joint_attach_kd = 100.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        if not self.no_grad:
            self.state = self.model.state(requires_grad=True)
            self.model.joint_q.requires_grad = True
            self.model.joint_qd.requires_grad = True
            self.model.joint_act.requires_grad = True
        else:
            self.state = self.model.state(requires_grad=False)

        start_joint_q, start_joint_qd, start_joint_act = self.get_state(return_act=True)
        # only stores a single copy of the initial state
        self.start_joint_q = start_joint_q.clone().view(self.num_envs, -1)[0]
        self.start_joint_qd = start_joint_qd.clone().view(self.num_envs, -1)[0]
        self.start_joint_act = start_joint_act.clone().view(self.num_envs, -1)[0]

    def render(self, mode="human"):
        if self.visualize:
            self.render_time += self.dt
            self.stage.begin_frame(self.render_time)
            self.stage.render(self.state)
            self.stage.end_frame()
            if self.num_frames == 40:
                self.stage.save()
                self.num_frames -= 40

    def step(self, actions):
        with wp.ScopedTimer("simulate", active=False, detailed=False):
            actions = torch.clip(actions, -1.0, 1.0)
            self.actions = actions.view(self.num_envs, -1)
            joint_act = self.action_strength * actions

            requires_grad = not self.no_grad
            if not self.no_grad:
                body_q = wp.to_torch(self.state.body_q)  # does this cut off grad to prev timestep?
                body_qd = wp.to_torch(self.state.body_qd)  # does this cut off grad to prev timestep?
                body_q.requires_grad = requires_grad
                body_qd.requires_grad = requires_grad
                assert self.model.body_q.requires_grad and self.state.body_q.requires_grad
                state_out = self.model.state(requires_grad=requires_grad)
                self.joint_q, self.joint_qd, self.state = IntegratorSimulate.apply(
                    self.model,
                    self.state,
                    self.integrator,
                    self.sim_dt,
                    self.sim_substeps,
                    joint_act,
                    body_q,
                    body_qd,
                    state_out,
                )
            else:
                for i in range(self.sim_substeps):
                    state_out = self.model.state(requires_grad=requires_grad)
                    self.state = self.integrator.simulate(self.model,
                                                          self.state, state_out,
                                                          self.sim_dt / float(self.sim_substeps)
                    )
                joint_q = wp.zeros_like(self.model.joint_q)
                joint_qd = wp.zeros_like(self.model.joint_qd)
                wp.sim.eval_ik(self.model, self.state, joint_q, joint_qd)
                self.joint_q, self.joint_qd = wp.to_torch(joint_q), wp.to_torch(joint_qd)

            self.sim_time += self.sim_dt

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                "obs_before_reset": self.obs_buf_before_reset,
                "episode_end": self.termination_buf,
            }

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        with wp.ScopedTimer("reset", active=False, detailed=False):
            if len(env_ids) > 0:
                self.reset(env_ids)

        with wp.ScopedTimer("render", active=False, detailed=False):
            self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        rand_init_q = np.pi * (
                    torch.rand(
                        size=(len(env_ids), self.num_joint_q), device=self.device
                    )
                    - 0.5
                )
        rand_init_qd = 0.5 * (
                    torch.rand(
                        size=(len(env_ids), self.num_joint_qd), device=self.device
                    )
                    - 0.5
                )
        joint_q[env_ids] += rand_init_q
        joint_qd[env_ids] += rand_init_qd
        return joint_q, joint_qd

    def initialize_trajectory(self):
        """ initialize_trajectory() starts collecting a new trajectory from the current states but cut off the computation graph to the previous states.
        It has to be called every time the algorithm starts an episode and return the observation vectors
        """
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def calculateObservations(self):
        if self.joint_q is None:
            self.joint_q, self.joint_qd = self.get_state()
            if not self.no_grad:
                self.joint_q.requires_grad = True
                self.joint_qd.requires_grad = True
                self.model.joint_act.requires_grad = True
                self.model.body_q.requires_grad = True
                self.model.body_qd.requires_grad = True
                self.state.body_q.requires_grad = True
                self.state.body_qd.requires_grad = True
        joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(self.num_envs, -1)

        x = joint_q[:, 0:1]
        theta = joint_q[:, 1:2]
        xdot = joint_qd[:, 0:1]
        theta_dot = joint_qd[:, 1:2]

        # observations: [x, xdot, sin(theta), cos(theta), theta_dot]
        self.obs_buf = torch.cat(
            [x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1
        )

    def calculateReward(self):
        joint_q = self.joint_q.view(self.num_envs, -1)
        joint_qd = self.joint_qd.view(self.num_envs, -1)

        x = joint_q[:, 0]
        theta = tu.normalize_angle(joint_q[:, 1])
        xdot = joint_qd[:, 0]
        theta_dot = joint_qd[:, 1]

        self.rew_buf = (
            -torch.pow(theta, 2.0) * self.pole_angle_penalty
            - torch.pow(theta_dot, 2.0) * self.pole_velocity_penalty
            - torch.pow(x, 2.0) * self.cart_position_penalty
            - torch.pow(xdot, 2.0) * self.cart_velocity_penalty
            - torch.sum(self.actions**2, dim=-1) * self.cart_action_penalty
        )

        # reset agents
        self.reset_buf = torch.where(
            self.progress_buf > self.episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
