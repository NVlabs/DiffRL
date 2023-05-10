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
from typing import List
from tqdm import tqdm, trange

import warp as wp
import warp.sim as sim
import warp.sim.render

wp.init()


class Bounce:
    # control frequency
    frame_dt = 1.0 / 60.0

    sim_time = 0.0
    render_time = 0.0
    sim_substeps = 8

    train_iters = 250
    train_rate = 0.01

    def __init__(
        self,
        num_envs=32,
        num_steps=200,
        start_state=np.array([-0.5, 1.0, 0.0]),
        start_vel=np.array([5.0, -5.0, 0.0]),
        render=False,
        profile=False,
        adapter=None,
        soft_contact_ke=1e4,  # stiffness
        soft_contact_kf=1e0,  # stiffness of friction
        soft_contact_kd=1e1,  # damping
        soft_contact_mu=0.9,  # friction coefficient
        soft_contact_margin=1e1,
        std=1e-1,
    ):
        self.device = wp.get_device(adapter)
        self.profile = profile
        self.render = render
        self.num_envs = num_envs

        self.start_state = start_state
        self.start_vel = start_vel
        self.std = std

        # sim frequency
        self.frame_steps = num_steps
        self.sim_steps = self.frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        noise = self.noise()
        builder: sim.ModelBuilder = wp.sim.ModelBuilder()
        for i in range(self.num_envs):
            builder.add_particle(pos=start_state, vel=start_vel + noise[i], mass=1.0)

        # for a large number of environments
        builder.soft_contact_max = 256 * 1024

        # high wall
        builder.add_shape_box(body=-1, pos=(2.0, 0.65, 0.0), hx=0.25, hy=0.65, hz=0.5)

        # short wall
        # builder.add_shape_box(body=-1, pos=(2.0, 0.5, 0.0), hx=0.25, hy=0.5, hz=0.5)

        self.model: sim.Model = builder.finalize(self.device)
        self.model.ground = True

        self.model.soft_contact_ke = soft_contact_ke
        self.model.soft_contact_kf = soft_contact_kf
        self.model.soft_contact_kd = soft_contact_kd
        self.model.soft_contact_mu = soft_contact_mu
        self.model.soft_contact_margin = soft_contact_margin

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.target = (-2.0, 1.5, 0.0)
        self.loss = wp.zeros(
            self.num_envs, dtype=wp.float32, device=self.device, requires_grad=True
        )
        self.l = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)

        # allocate sim states for trajectory
        self.states: List[wp.sim.State] = []
        self.contact_count = wp.zeros(self.sim_steps, dtype=int, device=self.device)
        self.prev_contact_count = np.zeros(self.sim_steps)
        for i in range(self.sim_steps + 1):
            self.states.append(self.model.state(requires_grad=True))

        if self.render:
            self.stage = wp.sim.render.SimRenderer(
                self.model,
                os.path.join(
                    os.path.dirname(__file__), "outputs/example_sim_grad_bounce.usd"
                ),
                scaling=40.0,
            )

    def noise(self):
        noise = np.random.normal(0.0, self.std, (self.num_envs, 2))
        noise[0] = 0.0  # for baseline
        self.noise_ = np.append(noise, np.zeros((self.num_envs, 1)), axis=1)
        return self.noise_

    def reset(self, start_state=None, start_vel=None):
        if start_state is None:
            start_state = self.start_state

        if start_vel is None:
            start_vel = self.start_vel

        # replicate array if necessary and assign
        q = np.array(start_state)
        if q.ndim == 1:
            q = np.tile(q, (self.num_envs, 1))
        q = wp.array(q, dtype=wp.vec3)
        self.model.particle_q.assign(q)

        # replicate array if necessary and assign
        qd = np.array(start_vel)
        if qd.ndim == 1:
            qd = np.tile(qd, (self.num_envs, 1))
        qd += self.noise()
        qd = wp.array(qd, dtype=wp.vec3)
        self.model.particle_qd.assign(qd)

        # only need to reset first state as everything is forward simulated from it
        self.states[0] = self.model.state(requires_grad=True)

        self.loss.zero_()
        self.l.zero_()

    @wp.kernel
    def loss_kernel(
        pos: wp.array(dtype=wp.vec3),
        target: wp.vec3,
        loss: wp.array(dtype=float),
    ):
        i = wp.tid()  # gets current thread id
        delta = pos[i] - target
        loss[i] = wp.dot(delta, delta)

    @wp.kernel
    def sum_kernel(
        losses: wp.array(dtype=float),
        loss: wp.array(dtype=float),
    ):
        i = wp.tid()
        wp.atomic_add(loss, 0, losses[i])

    @wp.kernel
    def mean_kernel(
        x_in: wp.array(dtype=wp.vec3), x_out: wp.array(dtype=wp.vec3), num_envs: float
    ):
        i = wp.tid()
        wp.atomic_add(x_out, 0, x_in[i] / num_envs)

    @wp.kernel
    def step_kernel(
        x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float
    ):
        tid = wp.tid()  # gets current thread id

        # gradient descent step
        x[tid] = x[tid] - grad[0] * alpha

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
            dim=self.num_envs,
            inputs=[self.model.soft_contact_count, i],
            outputs=[self.contact_count],
            device=self.device,
        )

    def compute_loss(self):
        # run control loop
        for i in range(self.sim_steps):
            self.states[i].clear_forces()
            wp.sim.collide(self.model, self.states[i])
            self.integrator.simulate(
                self.model, self.states[i], self.states[i + 1], self.sim_dt
            )
            self.count_contact_changes(i)

        # compute loss on final state
        wp.launch(
            self.loss_kernel,
            dim=self.num_envs,
            inputs=[self.states[-1].particle_q, self.target, self.loss],
            device=self.device,
        )

        return self.loss

    def sum_loss(self):
        wp.launch(
            self.sum_kernel,
            dim=self.num_envs,
            inputs=[self.loss, self.l],
            device=self.device,
        )
        return self.l

    def render_iter(self, iter):
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

        tape.backward(l)

        x_grad_analytic = tape.gradients[param]

        print(f"numeric grad: {x_grad_numeric}")
        print(f"analytic grad: {x_grad_analytic}")

        if self.render:
            self.render_iter(0)

        tape.zero()

    def trajectory(self):
        xy = []
        for s in self.states:
            pos = s.particle_q.numpy()[:, :2]
            vel = s.particle_qd.numpy()[:, :2]
            xy.append(np.concatenate((pos, vel), axis=-1))
        return np.array(xy)

    def train(
        self,
        iters,
        clip=False,
        norm=False,
        zero_order=False,
        tol=1e-4,
    ):
        losses = []
        trajectories = []
        grad_norms = []
        with trange(iters) as t:
            for i in t:
                tape = wp.Tape()

                with wp.ScopedTimer("Forward", active=self.profile):
                    with tape:
                        self.compute_loss()
                        self.sum_loss()

                with wp.ScopedTimer("Backward", active=self.profile):
                    if not zero_order:
                        tape.backward(self.l)

                if self.render:
                    with wp.ScopedTimer("Render", active=self.profile):
                        self.render_iter(i)

                with wp.ScopedTimer("Step", active=self.profile):
                    # need to take the mean of first qd
                    x = self.states[0].particle_qd
                    x = x.numpy()
                    x = x.mean(axis=0)

                    # need to take the mean of gradients
                    if zero_order:
                        loss = self.loss.numpy()
                        baseline = loss[0]
                        x_grad = (
                            1
                            / self.std**2
                            * (loss[..., None] - baseline)
                            * self.noise_
                        )
                        x_grad = x_grad.mean(axis=0)
                    else:
                        x_grad = tape.gradients[self.states[0].particle_qd]
                        x_grad = x_grad.numpy()
                        x_grad = x_grad.mean(axis=0)

                    if clip:
                        x_grad = np.clip(x_grad, -clip, clip)

                    if norm:
                        x_grad = norm * x_grad / np.linalg.norm(x_grad)

                    losses.append(self.loss.numpy())
                    trajectories.append(self.trajectory())

                    grad_norm = np.linalg.norm(x_grad)
                    grad_norms.append(grad_norm)

                    t.set_postfix(
                        loss=self.l.numpy()[0].round(4) / self.num_envs,
                        grad_norm=grad_norm,
                    )

                    # apply it to the initial state
                    x = x - x_grad * self.train_rate

                    # then add noise again and set to environments
                    x = x + self.noise()
                    self.states[0].particle_qd = wp.from_numpy(
                        x, dtype=wp.vec3, requires_grad=True
                    )

                # clear grad and loss for next iteration
                tape.zero()
                self.loss = wp.zeros_like(self.loss, requires_grad=True)
                self.l = wp.zeros_like(self.l, requires_grad=True)

                # early stopping
                # if len(losses) > 2:
                #     if np.abs(losses[-2].mean() - losses[-1].mean()) < tol:
                #         print("Early stopping at iter", i)
                #         break

            return np.array(losses), np.array(trajectories), np.array(grad_norms)

    def train_graph(self, iters, clip=False, norm=False, tol=1e-4):
        # capture forward/backward passes
        wp.capture_begin()

        tape = wp.Tape()
        with tape:
            self.compute_loss()
            self.sum_loss()

        tape.backward(self.l)

        self.graph = wp.capture_end()

        # replay and optimize
        losses = []
        trajectories = []
        last_l = 0.0

        for i in range(iters):
            with wp.ScopedTimer("Step", active=self.profile):
                # forward + backward
                wp.capture_launch(self.graph)
                # # count number of contact changes:
                # contact_count = self.contact_count.numpy()
                # contact_changes = np.sum(
                #     np.abs(contact_count - self.prev_contact_count)
                # )
                # self.prev_contact_count = contact_count
                # self.contact_count.zero_()

                # gradient descent step
                x = self.states[0].particle_qd
                self.x_grad = wp.zeros(1, dtype=wp.vec3)

                wp.launch(
                    self.mean_kernel,
                    dim=len(x),
                    inputs=[x.grad, self.x_grad, self.num_envs],
                    device=self.device,
                )

                print(f"Iter: {i} Loss: {self.l.numpy() - last_l}")
                print(f"Grad: {self.x_grad}")
                last_l = self.l.numpy()

                wp.launch(
                    self.step_kernel,
                    dim=len(x),
                    inputs=[x, self.x_grad, self.train_rate],
                    device=self.device,
                )

                # logging
                losses.append(self.loss.numpy())
                trajectories.append(self.trajectory())

                # clear grads for next iteration
                tape.zero()

            if len(losses) > 2:
                if np.abs(losses[-1].mean() - losses[-2].mean()) < tol:
                    print("Early stopping at iter", i)
                    break

            if self.render:
                with wp.ScopedTimer("Render", active=self.profile):
                    self.render_iter(i)

        return np.array(losses), np.array(trajectories)
