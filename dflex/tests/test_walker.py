# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import torch
import time

# include parent path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

from pxr import Usd, UsdGeom, Gf


class Walker:

    sim_duration = 5.0       # seconds
    sim_substeps = 16
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    render_time = 0.0

    train_iters = 50
    train_rate = 0.0001

    def __init__(self, mode="walker", adapter='cpu'):

        self.phase_count = 8
        self.phase_step = math.pi / self.phase_count * 2.0
        self.phase_freq = 20.0

        torch.manual_seed(42)

        builder = df.sim.ModelBuilder()

        walker = Usd.Stage.Open("assets/walker.usda")
        mesh = UsdGeom.Mesh(walker.GetPrimAtPath("/Grid/Grid"))

        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()

        for p in points:
            builder.add_particle(tuple(p), (0.0, 0.0, 0.0), 1.0)

        for t in range(0, len(indices), 3):
            i = indices[t + 0]
            j = indices[t + 1]
            k = indices[t + 2]

            builder.add_triangle(i, j, k)

        self.model = builder.finalize(adapter)
        self.model.tri_ke = 10000.0
        self.model.tri_ka = 10000.0
        self.model.tri_kd = 100.0
        self.model.tri_lift = 0.0
        self.model.tri_drag = 0.0

        self.edge_ke = 0.0
        self.edge_kd = 0.0

        self.model.contact_ke = 1.e+4
        self.model.contact_kd = 1000.0
        self.model.contact_kf = 1000.0
        self.model.contact_mu = 0.5

        self.model.particle_radius = 0.01

        # one fully connected layer + tanh activation
        self.network = torch.nn.Sequential(torch.nn.Linear(self.phase_count, self.model.tri_count, bias=False), torch.nn.Tanh()).to(adapter)

        self.activation_strength = 0.2
        self.activation_penalty = 0.1

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/walker.usd")

        self.renderer = df.render.UsdRenderer(self.model, self.stage)
        self.renderer.draw_points = True
        self.renderer.draw_springs = True
        self.renderer.draw_shapes = True

        self.integrator = df.sim.SemiImplicitIntegrator()

    def loss(self, render=True):

        #-----------------------
        # run simulation
        self.sim_time = 0.0

        self.state = self.model.state()

        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        for i in range(0, self.sim_steps):
            phases = torch.zeros(self.phase_count, device=self.model.adapter)

            # build sinusoidal phase inputs
            for p in range(self.phase_count):
                phases[p] = math.cos(4.0*self.sim_time*math.pi/(2.0*self.phase_count)*(2.0*p + 1.0))     #self.phase_freq*self.sim_time + p * self.phase_step)

            self.model.tri_activations = self.network(phases) * self.activation_strength
            self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

            self.sim_time += self.sim_dt

            if (render and (i % self.sim_substeps == 0)):
                self.render_time += self.sim_dt * self.sim_substeps
                self.renderer.update(self.state, self.render_time)

            com_pos = torch.mean(self.state.particle_q, 0)
            com_vel = torch.mean(self.state.particle_qd, 0)

            # use integral of velocity over course of the run
            loss = loss - com_vel[0] + torch.norm(self.model.tri_activations) * self.activation_penalty

        return loss

    def run(self):

        l = self.loss()
        self.stage.Save()

    def train(self, mode='gd'):

        # param to train
        self.step_count = 0
        render_freq = 1

        optimizer = None

        def closure():

            optimizer.zero_grad()

            # render every N steps
            render = False
            if ((self.step_count % render_freq) == 0):
                render = True

            l = self.loss(render)
            l.backward()

            print(str(self.step_count) + ": " + str(l))
            self.step_count += 1

            try:
                if (render):
                    self.stage.Save()
            except:
                print("USD save error")

            return l

        if (mode == 'gd'):

            # simple Gradient Descent
            for i in range(self.train_iters):

                closure()

                with torch.no_grad():
                    param -= self.train_rate * param.grad
        else:

            # L-BFGS
            if (mode == 'lbfgs'):
                optimizer = torch.optim.LBFGS(self.network.parameters(), lr=0.1, tolerance_grad=1.e-5, tolerance_change=0.01, line_search_fn="strong_wolfe")

            # Adam
            if (mode == 'adam'):
                optimizer = torch.optim.Adam(self.network.parameters(), lr=self.train_rate)

            # SGD
            if (mode == 'sgd'):
                optimizer = torch.optim.SGD(self.network.parameters(), lr=self.train_rate, momentum=0.25)

            # train
            for i in range(self.train_iters):
                optimizer.step(closure)

            # final save
            try:
                if (render):
                    self.stage.Save()
            except:
                print("USD save error")

    def save(self, file):
        torch.save(self.network, file)

    def load(self, file):
        self.network = torch.load(file)
        self.network.eval()


#---------

walker = Walker(adapter='cpu')
walker.train('lbfgs')
#walker.run()
