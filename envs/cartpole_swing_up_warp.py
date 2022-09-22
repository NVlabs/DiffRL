# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from envs.warp_env import WarpEnv
import math
import torch
import pdb

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warp as wp

import numpy as np

np.set_printoptions(precision=5, linewidth=256, suppress=True)

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

from utils import torch_utils as tu


@wp.kernel
def assign_kernel(
    b: wp.array(dtype=float),
    # outputs
    a: wp.array(dtype=float),
):
    tid = wp.tid()
    a[tid] = b[tid]


def float_assign(a, b):
    wp.launch(
        assign_kernel,
        dim=len(b),
        device=b.device,
        inputs=[b],
        outputs=[a],
    )
    return a


@wp.kernel
def assign_kernel_2d(
    b: wp.array2d(dtype=float),
    # outputs
    a: wp.array(dtype=float),
):
    tid = wp.tid()
    a[2 * tid] = b[tid, 0]
    a[2 * tid + 1] = b[tid, 1]


def float_assign_2d(a, b):
    wp.launch(
        assign_kernel_2d,
        dim=len(b),
        device=b.device,
        inputs=[b],
        outputs=[a],
    )
    return a


@wp.kernel
def assign_act_kernel(
    b: wp.array2d(dtype=float),
    # outputs
    a: wp.array(dtype=float),
):
    tid = wp.tid()
    a[2 * tid] = b[tid, 0]


def float_assign_joint_act(a, b):
    wp.launch(
        assign_act_kernel,
        dim=len(b),
        device=b.device,
        inputs=[b],
        outputs=[a],
    )
    return a


class IntegratorSimulate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        act,
        joint_q_start,
        joint_qd_start,
        # joint_q_end,
        # joint_qd_end,
        model,
        state_in,
        integrator,
        sim_dt,
        sim_substeps,
    ):
        ctx.tape = wp.Tape()
        ctx.model = model
        ctx.act = wp.from_torch(act)
        ctx.joint_q_start = wp.from_torch(joint_q_start)
        ctx.joint_qd_start = wp.from_torch(joint_qd_start)
        ctx.act.requires_grad = True
        # allocate output
        ctx.state_out = model.state(requires_grad=True)

        with ctx.tape:
            float_assign_joint_act(ctx.model.joint_act, ctx.act)
            float_assign_2d(ctx.model.joint_q, ctx.joint_q_start)
            float_assign_2d(ctx.model.joint_qd, ctx.joint_qd_start)
            # updates body position/vel
            for _ in range(sim_substeps):
                state_in = ctx.state_out = integrator.simulate(
                    model, state_in, ctx.state_out, sim_dt / float(sim_substeps)
                )
            # updates joint_q joint_qd
            ctx.joint_q_end, ctx.joint_qd_end = ctx.model.joint_q, ctx.model.joint_qd
            wp.sim.eval_ik(ctx.model, ctx.state_out, ctx.joint_q_end, ctx.joint_qd_end)

        return (
            wp.to_torch(ctx.joint_q_end).view(-1, 2),
            wp.to_torch(ctx.joint_qd_end).view(-1, 2),
        )

    @staticmethod
    def backward(ctx, adj_joint_q, adj_joint_qd):

        # map incoming Torch grads to our output variables
        ctx.joint_q_end.grad = wp.from_torch(adj_joint_q.flatten())
        ctx.joint_qd_end.grad = wp.from_torch(adj_joint_qd.flatten())

        ctx.tape.backward()
        joint_act_grad = wp.to_torch(ctx.tape.gradients[ctx.act])
        joint_q_grad = wp.to_torch(ctx.tape.gradients[ctx.joint_q_start])
        joint_qd_grad = wp.to_torch(ctx.tape.gradients[ctx.joint_qd_start])
        # print(f"joint_q_grad.norm(): {joint_q_grad.norm()}")
        # print(f"joint_qd_grad.norm(): {joint_qd_grad.norm()}")
        # print(f"joint_act_grad.norm(): {joint_act_grad.norm()}")

        # return adjoint w.r.t. inputs
        return (
            joint_act_grad,
            joint_q_grad,
            joint_qd_grad,
            None,
            None,
            None,
            None,
            None,
        )


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
            num_envs, num_obs, num_act, episode_length, seed, render, device
        )

        self.no_grad = no_grad
        self.stochastic_init = stochastic_init
        self.early_termination = early_termination

        self.init_sim()

        # action parameters
        self.action_strength = 1000.0

        # loss related
        self.pole_angle_penalty = 1.0
        self.pole_velocity_penalty = 0.1

        self.cart_position_penalty = 0.05
        self.cart_velocity_penalty = 0.1

        self.cart_action_penalty = 0.0

        # -----------------------
        # set up Usd renderer
        if self.visualize:
            self.stage = wp.sim.render.SimRenderer(
                self.model,
                os.path.join(
                    os.path.dirname(__file__),
                    f"outputs/CartPoleSwingUp_{self.num_envs}.usd",
                ),
            )
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
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e4,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
        )

        self.builder = wp.sim.ModelBuilder()

        for i in range(self.num_envs):
            self.builder.add_rigid_articulation(
                self.articulation_builder,
                xform=wp.transform(
                    np.array((i * 2.0, 4.0, 0.0)),
                    wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
                ),
            )
            self.builder.joint_q[i * self.num_joint_q + 1] = -math.pi

        self.model = self.builder.finalize(str(self.device))
        self.model.ground = True

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.state = self.model.state(requires_grad=True)
        self.model.joint_q.requires_grad = True
        self.model.joint_qd.requires_grad = True
        self.model.joint_act.requires_grad = True
        self.state.body_q.requires_grad = True
        self.state.body_qd.requires_grad = True

        start_joint_q, start_joint_qd, start_joint_act = self.get_state(return_act=True)
        self.start_joint_q = start_joint_q.detach().view(self.num_envs, -1)
        self.start_joint_qd = start_joint_qd.detach().view(self.num_envs, -1)
        self.start_joint_act = start_joint_act.detach().view(self.num_envs, -1)

    def render(self, mode="human"):
        if self.visualize:
            self.render_time += self.dt
            self.stage.update(self.state, self.render_time)
            if self.num_frames == 40:
                try:
                    self.stage.Save()
                except:
                    print("USD save error")
                self.num_frames -= 40

    def step(self, actions):
        with wp.ScopedTimer("simulate", active=False, detailed=False):
            actions = torch.clip(actions, -1.0, 1.0)
            self.actions = actions.view(self.num_envs, -1)
            joint_act = self.action_strength * actions

            self.state.clear_forces()

            self.joint_q, self.joint_qd = IntegratorSimulate.apply(
                joint_act.detach(),
                self.joint_q,
                self.joint_qd,
                self.model,
                self.state,
                self.integrator,
                self.sim_dt,
                self.sim_substeps,
            )
            self.state = self.model.state()

            self.sim_time += self.sim_dt

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        # self.joint_q, self.joint_qd = self.joint_q_next, self.joint_qd_next
        self.calculateObservations()
        self.calculateReward()

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.detach()
            self.extras = {
                "obs_before_reset": self.obs_buf_before_reset,
                "episode_end": self.termination_buf,
            }

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # self.obs_buf_before_reset = self.obs_buf.clone()

        with wp.ScopedTimer("reset", active=False, detailed=False):
            if len(env_ids) > 0:
                self.reset(env_ids)

        with wp.ScopedTimer("render", active=False, detailed=False):
            self.render()

        # self.extras = {'obs_before_reset': self.obs_buf_before_reset}

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(
                    self.num_envs, dtype=torch.long, device=self.device
                )
        if env_ids is not None:
            # fixed start state
            self.joint_q, self.joint_qd, joint_act = self.get_state(return_act=True)
            self.joint_q[env_ids] = self.start_joint_q[env_ids].clone()
            self.joint_qd[env_ids] = self.start_joint_qd[env_ids].clone()
            joint_act[env_ids] = self.start_joint_act[env_ids].clone()

            if self.stochastic_init:
                self.joint_q[env_ids] += np.pi * (
                    torch.rand(
                        size=(len(env_ids), self.num_joint_q), device=self.device
                    )
                    - 0.5
                )
                self.joint_qd[env_ids] += 0.5 * (
                    torch.rand(
                        size=(len(env_ids), self.num_joint_qd), device=self.device
                    )
                    - 0.5
                )
            self.joint_q.requires_grad = True
            self.joint_qd.requires_grad = True
            self.model.joint_q.assign(wp.from_torch(self.joint_q.flatten()))
            self.model.joint_qd.assign(wp.from_torch(self.joint_qd.flatten()))
            self.model.joint_qd.assign(wp.from_torch(joint_act.flatten()))

            self.progress_buf[env_ids] = 0

            self.calculateObservations()

        return self.obs_buf

    """
    cut off the gradient from the current state to previous states
    """

    def clear_grad(self):
        with torch.no_grad():  # TODO: check with Miles
            current_joint_q = wp.to_torch(self.model.joint_q).detach()
            current_joint_qd = wp.to_torch(self.model.joint_qd).detach()
            current_joint_act = wp.to_torch(self.model.joint_act).detach()
            self.state = self.model.state(requires_grad=True)
            self.model.joint_q.assign(wp.from_torch(current_joint_q))
            self.model.joint_qd.assign(wp.from_torch(current_joint_qd))
            self.model.joint_act.assign(wp.from_torch(current_joint_act))
            self.model.joint_q.grad.zero_()
            self.model.joint_qd.grad.zero_()
            self.model.joint_act.grad.zero_()
            self.joint_q = None

    """
    This function starts collecting a new trajectory from the current states but cut off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and return the observation vectors
    """

    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def calculateObservations(self):
        if self.joint_q is None:
            self.joint_q, self.joint_qd = self.get_state()
            self.joint_q.requires_grad = True
            self.joint_qd.requires_grad = True
        x = self.joint_q[:, 0:1]
        theta = self.joint_q[:, 1:2]
        xdot = self.joint_qd[:, 0:1]
        theta_dot = self.joint_qd[:, 1:2]

        # observations: [x, xdot, sin(theta), cos(theta), theta_dot]
        self.obs_buf = torch.cat(
            [x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1
        )

    def calculateReward(self):
        x = self.joint_q[:, 0]
        theta = tu.normalize_angle(self.joint_q[:, 1])
        xdot = self.joint_qd[:, 0]
        theta_dot = self.joint_qd[:, 1]

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
