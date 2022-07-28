# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import time

# include parent path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

from pxr import Usd, UsdGeom, Gf


class Cage:

    sim_duration = 2.0       # seconds
    sim_substeps = 8
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 20
    train_rate = 0.1         #1.0/(sim_dt*sim_dt)

    def __init__(self, mode="quad", adapter='cpu'):

        builder = df.sim.ModelBuilder()

        if (mode == "quad"):

            # anchors
            builder.add_particle((-1.0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)
            builder.add_particle((1.0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)
            builder.add_particle((1.0, -1.0, 0.0), (0.0, 0.0, 0.0), 0.0)
            builder.add_particle((-1.0, -1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

            # ball
            builder.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)

            ke = 1.e+2
            kd = 10.0

            # springs
            builder.add_spring(0, 4, ke, kd, 0)
            builder.add_spring(1, 4, ke, kd, 0)
            builder.add_spring(2, 4, ke, kd, 0)
            builder.add_spring(3, 4, ke, kd, 0)

            self.target_pos = torch.tensor((0.85, 0.5, 0.0), device=adapter)
            self.target_index = 4

        if (mode == "box"):

            # anchors
            builder.add_particle((-1.0, -1.0, -1.0), (0.0, 0.0, 0.0), 0.0)
            builder.add_particle((-1.0, -1.0, 1.0), (0.0, 0.0, 0.0), 0.0)
            builder.add_particle((-1.0, 1.0, -1.0), (0.0, 0.0, 0.0), 0.0)
            builder.add_particle((-1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.0)
            builder.add_particle((1.0, -1.0, -1.0), (0.0, 0.0, 0.0), 0.0)
            builder.add_particle((1.0, -1.0, 1.0), (0.0, 0.0, 0.0), 0.0)
            builder.add_particle((1.0, 1.0, -1.0), (0.0, 0.0, 0.0), 0.0)
            builder.add_particle((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.0)

            # ball
            builder.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)

            ke = 1.e+2
            kd = 10.0

            target = 8

            # springs
            builder.add_spring(0, target, ke, kd, 0)
            builder.add_spring(1, target, ke, kd, 0)
            builder.add_spring(2, target, ke, kd, 0)
            builder.add_spring(3, target, ke, kd, 0)
            builder.add_spring(4, target, ke, kd, 0)
            builder.add_spring(5, target, ke, kd, 0)
            builder.add_spring(6, target, ke, kd, 0)
            builder.add_spring(7, target, ke, kd, 0)

            self.target_pos = torch.tensor((0.85, 0.5, -0.75), device=adapter)
            self.target_index = target

        if (mode == "chain"):

            # anchor
            builder.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)

            segments = 4
            segment_length = 1.0

            ke = 1.e+2
            kd = 10.0

            for i in range(1, segments + 1):

                builder.add_particle((segment_length * i, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)
                builder.add_spring(i - 1, i, ke, kd, 0)

                # bending spring
                if (i > 1):
                    builder.add_spring(i - 2, i, ke * 4.0, kd, 0)

            self.target_pos = torch.tensor((3.0, 0.0, 0.0), device=adapter)
            self.target_index = segments

        self.model = builder.finalize(adapter)
        self.model.particle_radius = 0.05
        self.model.ground = False
        self.model.gravity = torch.tensor((0.0, 0.0, 0.0), device=adapter)

        # set optimization targets
        self.model.spring_rest_length.requires_grad_()

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/cage.usda")

        self.renderer = df.render.UsdRenderer(self.model, self.stage)
        self.renderer.draw_points = True
        self.renderer.draw_springs = True
        self.renderer.draw_shapes = True

        self.renderer.add_sphere(self.target_pos.tolist(), 0.1, "target")

        self.integrator = df.sim.SemiImplicitIntegrator()

    def loss(self):

        #-----------------------
        # run simulation

        self.state = self.model.state()

        for i in range(0, self.sim_steps):

            self.state = self.integrator.forward(self.model, self.state, self.sim_dt)
            # print("state: ", self.state.particle_q[self.target_index])

            if (i % self.sim_substeps) == 0:
                self.renderer.update(self.state, self.sim_time)

            self.sim_time += self.sim_dt

        # print(self.state.particle_q[self.target_index])
        loss = torch.norm(self.state.particle_q[self.target_index] - self.target_pos)

        return loss

    def run(self):

        l = self.loss()
        self.stage.Save()

    def train(self, mode='gd'):

        # param to train
        param = self.model.spring_rest_length

        # Gradient Descent
        if (mode == 'gd'):
            for i in range(self.train_iters):
                # with torch.autograd.detect_anomaly():
                l = self.loss()
                print(l)
                l.backward()

                with torch.no_grad():
                    param -= self.train_rate * param.grad
                    param.grad.zero_()

        # L-BFGS
        if (mode == 'lbfgs'):

            optimizer = torch.optim.LBFGS([param], self.train_rate, tolerance_grad=1.e-5, history_size=4, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                l = self.loss()
                l.backward()

                print(l)

                return l

            for i in range(self.train_iters):
                optimizer.step(closure)

        # SGD
        if (mode == 'sgd'):

            optimizer = torch.optim.SGD([param], lr=self.train_rate, momentum=0.8)

            for i in range(self.train_iters):
                optimizer.zero_grad()

                l = self.loss()
                l.backward()

                print(l)

                optimizer.step()

        self.stage.Save()


#---------

cage = Cage("box", adapter='cpu')
cage.train('gd')
#cage.run()
