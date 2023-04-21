# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Grad Bounce
#
# Shows how to use Warp to optimize the initial velocity of a particle
# such that it bounces off the wall and floor in order to hit a target.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the initial velocity, followed by
# a simple gradient-descent optimization step.
#
###########################################################################

import os

import numpy as np
import torch
from tqdm import tqdm

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Bounce:
    # seconds
    sim_duration = 0.6

    # control frequency
    frame_dt = 1.0 / 60.0
    frame_steps = int(sim_duration / frame_dt)

    # sim frequency
    sim_substeps = 8
    sim_steps = frame_steps * sim_substeps
    sim_dt = frame_dt / sim_substeps
    print("H=", sim_steps)

    sim_time = 0.0
    render_time = 0.0

    start_vel = (5.0, -5.0, 0.0)

    def __init__(
        self,
        render=True,
        profile=False,
        adapter=None,
        num_envs=1024,
        start_state=[-0.5, 1.0, 5.0, -5.0],
        target=[2.0, 1.0, 0.0],
    ):
        self.num_envs = num_envs
        self.start_state = np.array(start_state)
        self.target = np.array(target)

        builder = wp.sim.ModelBuilder()

        for i in range(self.num_envs):
            builder.add_particle(
                pos=(-self.start_state[0], self.start_state[1], 0.0),
                vel=(self.start_state[2], self.start_state[3], 0.0),
                mass=1.0,
            )
            builder.add_shape_box(
                body=-1,
                pos=(self.target[0], self.target[1], 0.0),
                hx=0.25,
                hy=1.0,
                hz=1.0,
            )

        self.device = wp.get_device(adapter)
        self.profile = profile

        self.model = builder.finalize(self.device)
        self.model.ground = True

        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kf = 0.0
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_mu = 0.25
        self.model.soft_contact_margin = 10.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        # self.target = (-2.0, 1.5, 0.0)
        self.loss = wp.zeros(
            1,
            dtype=wp.float32,
            device=self.device,
            requires_grad=True
            # self.num_envs, dtype=wp.float32, device=self.device, requires_grad=True
        )

        # allocate sim states for trajectory
        self.states = []
        self.contact_count = wp.zeros(self.sim_steps, dtype=int, device=self.device)
        self.prev_contact_count = np.zeros(self.sim_steps)
        for i in range(self.sim_steps + 1):
            self.states.append(self.model.state(requires_grad=True))

        # one-shot contact creation (valid if we're doing simple collision against a constant normal plane)
        wp.sim.collide(self.model, self.states[0])

        if self.render:
            self.stage = wp.sim.render.SimRenderer(
                self.model,
                os.path.join(
                    os.path.dirname(__file__), "outputs/example_sim_grad_bounce.usd"
                ),
                scaling=40.0,
            )

    @wp.kernel
    def loss_kernel(
        pos: wp.array(dtype=wp.vec3),
        target: wp.vec3,
        # target: wp.array(dtype=wp.vec3),
        loss: wp.array(dtype=float),
    ):
        # distance to target
        delta = pos[0] - target
        loss[0] = wp.dot(delta, delta)
        # delta = pos - target
        # loss = torch.norm(delta, 2, dim=-1)

    @wp.kernel
    def step_kernel(
        x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float
    ):
        tid = wp.tid()

        # gradient descent step
        x[tid] = x[tid] - grad[tid] * alpha

    @wp.kernel
    def count_contact_changes_kernel(
        contact_count: wp.array(dtype=int),
        contact_count_id: int,
        contact_count_copy: wp.array(dtype=int),
    ):
        wp.atomic_add(contact_count_copy, contact_count_id, contact_count[0])

    def count_contact_changes(self, i):
        # count contact changes
        wp.launch(
            self.count_contact_changes_kernel,
            dim=1,
            inputs=[self.model.soft_contact_count, i],
            outputs=[self.contact_count],
            device=self.device,
        )

    def compute_loss(self, v, H):
        # set starting state
        start_pos = np.array([self.start_state[0], self.start_state[1], 0.0])
        start_pos = np.repeat(start_pos.reshape((1, -1)), self.num_envs, axis=0)
        self.states[0].particle_q.assign(start_pos)

        # add 0 velocity if necessary
        if v.shape[1] == 2:
            v = np.append(v, np.zeros((self.num_envs, 1)), axis=1)

        assert v.shape == (self.num_envs, 3), v.shape
        self.states[0].particle_qd.assign(v)

        # run control loop
        for i in range(H):
            self.states[i].clear_forces()

            self.integrator.simulate(
                self.model, self.states[i], self.states[i + 1], self.sim_dt
            )
            self.count_contact_changes(i)

        # compute loss on final state
        wp.launch(
            self.loss_kernel,
            dim=1,
            inputs=[
                self.states[H].particle_q,
                self.target,
                self.loss,
            ],
            device=self.device,
        )
        # loss = torch.tensor(self.states[H].particle_q) - torch.tensor(self.target)
        # loss = torch.norm(loss, p=2, dim=-1)
        return self.loss

    def render(self, iter):
        # render every 16 iters
        if iter % 16 > 0:
            return

        # draw trajectory
        traj_verts = [self.states[0].particle_q.numpy()[0].tolist()]

        for i in range(0, self.sim_steps, self.sim_substeps):
            traj_verts.append(self.states[i].particle_q.numpy()[0].tolist())

            self.stage.begin_frame(self.render_time)
            self.stage.render(self.states[i])
            self.stage.render_box(
                pos=self.target,
                rot=wp.quat_identity(),
                extents=(0.1, 0.1, 0.1),
                name="target",
            )
            self.stage.render_line_strip(
                vertices=traj_verts,
                color=wp.render.bourke_color_map(0.0, 7.0, self.loss.numpy()[0]),
                radius=0.02,
                name=f"traj_{iter}",
            )
            self.stage.end_frame()

            self.render_time += self.frame_dt

        self.stage.save()

    def check_grad(self):
        param = self.states[0].particle_qd

        # initial value
        x_c = param.numpy().flatten()

        # compute numeric gradient
        x_grad_numeric = np.zeros_like(x_c)

        for i in range(len(x_c)):
            eps = 1.0e-3

            step = np.zeros_like(x_c)
            step[i] = eps

            x_1 = x_c + step
            x_0 = x_c - step

            param.assign(x_1)
            l_1 = self.compute_loss().numpy()[0]

            param.assign(x_0)
            l_0 = self.compute_loss().numpy()[0]

            dldx = (l_1 - l_0) / (eps * 2.0)

            x_grad_numeric[i] = dldx

        # reset initial state
        param.assign(x_c)

        # compute analytic gradient
        tape = wp.Tape()
        with tape:
            l = self.compute_loss()

        tape.sum().backward(l)

        x_grad_analytic = tape.gradients[param]

        print(f"numeric grad: {x_grad_numeric}")
        print(f"analytic grad: {x_grad_analytic}")

        tape.zero()


np.random.seed(0)
torch.manual_seed(0)

n = 2  # [x,y] position
m = 2  # [v_x, v_y] velocities
N = 1024
H = 200

# Crete environment
bounce = Bounce(profile=False, render=True, num_envs=N)

# create a random set of actions
std = 0.5
w = np.random.normal(0.0, std, (N, m))
w[0] = 0.0
fobgs = []
zobgs = []
losses = []
baseline = []

start = np.array([5.0, -5.0])

for h in tqdm(range(1, H + 1)):
    param = bounce.states[0].particle_qd
    tape = wp.Tape()
    with tape:
        loss = bounce.compute_loss(start + w, h)
    # loss = wp.array(loss.sum(), requires_grad=True)
    print("loss", loss)
    tape.backward(loss)
    fobgs = tape.gradients[param]
    print(fobgs)
    print("fobg shape", fobgs.shape)
    exit(0)

print(loss)
