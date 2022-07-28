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


class Contact:

    sim_duration = 2.0       # seconds
    sim_substeps = 16
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 16
    train_rate = 0.1         #1.0/(sim_dt*sim_dt)

    def __init__(self, adapter='cpu'):

        builder = df.sim.ModelBuilder()

        builder.add_particle((0.0, 1.5, 0.0), (0.0, 0.0, 0.0), 0.25)

        self.target_pos = torch.tensor((3.0, 0.0, 0.0), device=adapter)
        self.target_index = 0

        self.model = builder.finalize(adapter)
        self.model.contact_ke = 1.e+3
        self.model.contact_kf = 10.0
        self.model.contact_kd = 10.0
        self.model.contact_mu = 0.25

        self.model.particle_qd.requires_grad = True

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/contact.usda")

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

            if (i % self.sim_substeps) == 0:
                self.renderer.update(self.state, self.sim_time)

            self.sim_time += self.sim_dt

        self.stage.Save()

        loss = torch.norm(self.state.particle_q[self.target_index] - self.target_pos)
        return loss

    def run(self):

        l = self.loss()
        self.stage.Save()

    def train(self, mode='gd'):

        # param to train
        param = self.model.particle_qd

        # Gradient Descent
        if (mode == 'gd'):
            for i in range(self.train_iters):

                l = self.loss()
                l.backward()

                print("loss: " + str(l.item()))
                print("v: " + str(param))
                print("vgrad: " + str(param.grad))
                print("--------------------")

                with torch.no_grad():
                    param -= self.train_rate * param.grad
                    param.grad.zero_()

        # L-BFGS
        if (mode == 'lbfgs'):

            optimizer = torch.optim.LBFGS([param], 1.0, tolerance_grad=1.e-5, history_size=4, line_search_fn="strong_wolfe")

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
                print(param)

                optimizer.step()

        self.stage.Save()


#---------

contact = Contact(adapter='cpu')
contact.train('lbfgs')

