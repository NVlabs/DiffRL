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


class Chain:

    sim_duration = 10.0      # seconds
    sim_substeps = 2
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 20
    train_rate = 0.1         #1.0/(sim_dt*sim_dt)

    def __init__(self, adapter='cpu'):

        builder = df.sim.ModelBuilder()

        # anchor
        builder.add_particle((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

        for i in range(1, 10):
            builder.add_particle((i, 1.0, 0.0), (0.0, 0., 0.0), 1.0)
            builder.add_spring(i - 1, i, 1.e+6, 0.0, 0)

        self.model = builder.finalize(adapter)
        self.model.ground = False

        self.impulse = torch.tensor((0.0, 0.0, 0.0), requires_grad=True, device=adapter)

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/chain.usda")

        self.renderer = df.render.UsdRenderer(self.model, self.stage)
        self.renderer.draw_points = True
        self.renderer.draw_springs = True
        self.renderer.draw_shapes = True

        #self.integrator = df.sim.SemiImplicitIntegrator()
        self.integrator = df.sim.XPBDIntegrator()

    def loss(self):

        #-----------------------
        # run simulation

        self.state = self.model.state()
        self.state.particle_qd[1] = self.impulse

        for i in range(0, self.sim_steps):

            self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

            if (i % self.sim_substeps) == 0:
                self.renderer.update(self.state, self.sim_time)

            self.sim_time += self.sim_dt

        target = torch.tensor((0.0, 2.0, 0.0), device=self.model.adapter)

        loss = torch.norm(self.state.particle_q[1] - target)
        return loss

    def run(self):

        l = self.loss()
        self.stage.Save()

    def train(self, mode='gd'):

        # param to train
        param = self.impulse

        # Gradient Descent
        if (mode == 'gd'):
            for i in range(self.train_iters):

                l = self.loss()
                l.backward()

                print(l)

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

chain = Chain(adapter='cpu')
#cloth.train('lbfgs')
chain.run()
