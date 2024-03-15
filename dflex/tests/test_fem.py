# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import torch
import cProfile

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dflex as df

from pxr import Usd, UsdGeom, Gf


class FEM:
    sim_duration = 5.0  # seconds
    sim_substeps = 32
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 16
    train_rate = 0.01  # 1.0/(sim_dt*sim_dt)

    phase_count = 8
    phase_step = math.pi / phase_count * 2.0
    phase_freq = 2.5

    def __init__(self, adapter="cpu"):
        torch.manual_seed(42)

        builder = df.sim.ModelBuilder()

        mesh = Usd.Stage.Open("assets/prop.usda")
        geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/mesh"))

        points = geom.GetPointsAttr().Get()

        tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()
        tri_indices = geom.GetFaceVertexIndicesAttr().Get()
        tri_counts = geom.GetFaceVertexCountsAttr().Get()

        r = df.quat_multiply(
            df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0),
            df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.0),
        )

        builder.add_soft_mesh(
            pos=(0.0, 2.0, 0.0),
            rot=r,
            scale=1.0,
            vel=(1.5, 0.0, 0.0),
            vertices=points,
            indices=tet_indices,
            density=1.0,
            k_mu=1000.0,
            k_lambda=1000.0,
            k_damp=1.0,
        )

        # builder.add_soft_grid(pos=(0.0, 0.5, 0.0), rot=(0.0, 0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), dim_x=1, dim_y=2, dim_z=1, cell_x=0.5, cell_y=0.5, cell_z=0.5, density=1.0)

        # s = 2.0
        # builder.add_particle((0.0, 0.5, 0.0), (0.0, 0.0, 0.0), 1.0)
        # builder.add_particle((s,  0.5, 0.0), (0.0, 0.0, 0.0), 1.0)
        # builder.add_particle((0.0, 0.5, s), (0.0, 0.0, 0.0), 1.0)
        # builder.add_particle((0.0, s + 0.5, 0.0), (0.0, 0.0, 0.0), 1.0)

        # builder.add_tetrahedron(1, 3, 0, 2)

        self.model = builder.finalize(adapter)

        # self.model.tet_kl = 1000.0
        # self.model.tet_km = 1000.0
        # self.model.tet_kd = 1.0

        # disable triangle dynamics (just used for rendering)
        self.model.tri_ke = 0.0
        self.model.tri_ka = 0.0
        self.model.tri_kd = 0.0
        self.model.tri_kb = 0.0

        self.model.contact_ke = 1.0e4
        self.model.contact_kd = 1.0
        self.model.contact_kf = 10.0
        self.model.contact_mu = 0.5

        self.model.particle_radius = 0.05
        self.model.ground = True

        # one fully connected layer + tanh activation
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.phase_count, self.model.tet_count, bias=False),
            torch.nn.Tanh(),
        ).to(adapter)

        self.activation_strength = 0.3
        self.activation_penalty = 0.0

        # -----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/fem.usd")

        if self.stage:
            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

        self.integrator = df.sim.SemiImplicitIntegrator()

    def loss(self, render=True):
        # -----------------------
        # run simulation
        self.sim_time = 0.0

        self.state = self.model.state()

        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        for i in range(0, self.sim_steps):
            # build sinusoidal input phases
            with df.ScopedTimer("inference", False):
                phases = torch.zeros(self.phase_count, device=self.model.adapter)
                for p in range(self.phase_count):
                    phases[p] = math.sin(
                        self.phase_freq * self.sim_time + p * self.phase_step
                    )

                # compute activations (rest angles)
                self.model.tet_activations = (
                    self.network(phases) * self.activation_strength
                )

            # forward dynamics
            with df.ScopedTimer("simulate", False):
                self.state = self.integrator.forward(
                    self.model, self.state, self.sim_dt
                )
                self.sim_time += self.sim_dt

            # render
            with df.ScopedTimer("render", False):
                if self.stage and render and (i % self.sim_substeps == 0):
                    self.render_time += self.sim_dt * self.sim_substeps
                    self.renderer.update(self.state, self.render_time)

            # loss
            with df.ScopedTimer("loss", False):
                com_loss = torch.mean(self.state.particle_qd, 0)
                # act_loss = torch.norm(selfactivation)*self.activation_penalty

                loss = loss - com_loss[0]  # - act_loss

        return loss

    def run(self, profile=False, render=True):
        df.config.no_grad = True

        with torch.no_grad():
            with df.ScopedTimer("run"):
                if profile:
                    cp = cProfile.Profile()
                    cp.clear()
                    cp.enable()

                # run forward dynamics
                if profile:
                    self.state = self.model.state()
                    for i in range(0, self.sim_steps):
                        self.state = self.integrator.forward(
                            self.model, self.state, self.sim_dt
                        )
                        self.sim_time += self.sim_dt
                else:
                    l = self.loss(render)

                if profile:
                    cp.disable()
                    cp.print_stats(sort="tottime")

        if self.stage:
            self.stage.Save()

    def train(self, mode="gd"):
        # param to train
        self.step_count = 0
        render_freq = 1

        optimizer = None

        def closure():
            optimizer.zero_grad()

            # render every N steps
            render = False
            if (self.step_count % render_freq) == 0:
                render = True

            with df.ScopedTimer("forward"):
                # with torch.autograd.detect_anomaly():
                l = self.loss(render)

            with df.ScopedTimer("backward"):
                # with torch.autograd.detect_anomaly():
                l.backward()

            print(str(self.step_count) + ": " + str(l))
            self.step_count += 1

            with df.ScopedTimer("save"):
                try:
                    if render:
                        self.stage.Save()
                except:
                    print("USD save error")

            return l

        with df.ScopedTimer("step"):
            if mode == "gd":
                # simple Gradient Descent
                for i in range(self.train_iters):
                    closure()

                    with torch.no_grad():
                        param -= self.train_rate * param.grad
            else:
                # L-BFGS
                if mode == "lbfgs":
                    optimizer = torch.optim.LBFGS(
                        self.network.parameters(),
                        lr=1.0,
                        tolerance_grad=1.0e-5,
                        tolerance_change=0.01,
                        line_search_fn="strong_wolfe",
                    )

                # Adam
                if mode == "adam":
                    optimizer = torch.optim.Adam(
                        self.network.parameters(), lr=self.train_rate
                    )

                # SGD
                if mode == "sgd":
                    optimizer = torch.optim.SGD(
                        self.network.parameters(), lr=self.train_rate, momentum=0.5
                    )

                # train
                for i in range(self.train_iters):
                    optimizer.step(closure)

                # final save
                try:
                    if render:
                        self.stage.Save()
                except:
                    print("USD save error")

    def save(self, file):
        torch.save(self.network, file)

    def load(self, file):
        self.network = torch.load(file)
        self.network.eval()


# ---------

fem = FEM(adapter="cuda")
fem.run(profile=False, render=True)

# fem.train('lbfgs')
# fem.train('sgd')
