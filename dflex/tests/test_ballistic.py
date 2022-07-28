# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import time
import math

# include parent path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

from pxr import Usd, UsdGeom, Gf


class Ballistic:

    sim_duration = 2.0       # seconds
    sim_substeps = 10
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 5
    train_rate = 0.1         #1.0/(sim_dt*sim_dt)

    def __init__(self, adapter='cpu'):

        builder = df.sim.ModelBuilder()

        builder.add_particle((0, 1.0, 0.0), (0.1, 0.0, 0.0), 1.0)

        self.model = builder.finalize(adapter)

        self.target = torch.tensor((2.0, 1.0, 0.0), device=adapter)

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/ballistic.usda")

        self.renderer = df.render.UsdRenderer(self.model, self.stage)
        self.renderer.draw_points = True
        self.renderer.draw_springs = True
        self.renderer.draw_shapes = True

        self.renderer.add_sphere(self.target.tolist(), 0.1, "target")

        self.integrator = df.sim.SemiImplicitIntegrator()

    def loss(self):

        #-----------------------
        # run simulation

        self.state = self.model.state()

        for i in range(0, self.sim_steps):

            self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

            if (i % self.sim_substeps) == 0:
                self.renderer.update(self.state, self.sim_time)

            self.sim_time += self.sim_dt

        loss = torch.norm(self.state.particle_q[0] - self.target)
        return loss

    def train(self, mode='gd'):

        # Gradient Descent
        if (mode == 'gd'):
            for i in range(self.train_iters):

                l = self.loss()
                l.backward()

                print(l)

                with torch.no_grad():
                    self.model.particle_v -= self.train_rate * self.model.particle_v.grad
                    self.model.particle_v.grad.zero_()

        # L-BFGS
        if (mode == 'lbfgs'):

            optimizer = torch.optim.LBFGS([self.model.particle_v], self.train_rate, tolerance_grad=1.e-5, history_size=4, line_search_fn="strong_wolfe")

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

            optimizer = torch.optim.SGD([self.model.particle_v], lr=self.train_rate, momentum=0.8)

            for i in range(self.train_iters):
                optimizer.zero_grad()

                l = self.loss()
                l.backward()

                print(l)

                optimizer.step()

        self.stage.Save()


#---------

ballistic = Ballistic(adapter='cpu')
ballistic.train('lbfgs')
