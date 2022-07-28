# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

from pxr import Usd, UsdGeom, Gf


class Beam:

    sim_duration = 3.0       # seconds
    sim_substeps = 32
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 64
    train_rate = 1.0

    def __init__(self, device='cpu'):

        torch.manual_seed(42)

        builder = df.sim.ModelBuilder()
        builder.add_soft_grid(pos=(0.0, 0.0, 0.0),
                              rot=df.quat_identity(),
                              vel=(0.0, 0.0, 0.0),
                              dim_x=20,
                              dim_y=2,
                              dim_z=2,
                              cell_x=0.1,
                              cell_y=0.1,
                              cell_z=0.1,
                              density=10.0,
                              k_mu=1000.0,
                              k_lambda=1000.0,
                              k_damp=5.0,
                              fix_left=True,
                              fix_right=True)

        self.model = builder.finalize(device)

        # disable triangle dynamics (just used for rendering)
        self.model.tri_ke = 0.0
        self.model.tri_ka = 0.0
        self.model.tri_kd = 0.0
        self.model.tri_kb = 0.0

        self.model.particle_radius = 0.05
        self.model.ground = False

        self.target = torch.tensor((-0.5)).to(device)
        self.material = torch.tensor((100.0, 50.0, 5.0), requires_grad=True, device=device)

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/beam.usd")

        if (self.stage):
            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

        self.integrator = df.sim.SemiImplicitIntegrator()

    def loss(self, render=True):

        #-----------------------
        # run simulation
        self.sim_time = 0.0

        self.state = self.model.state()

        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        for i in range(0, self.sim_steps):

            # clamp material params to reasonable range
            mat_min = torch.tensor((1.e+1, 1.e+1, 5.0), device=self.model.adapter)
            mat_max = torch.tensor((1.e+5, 1.e+5, 5.0), device=self.model.adapter)
            mat_val = torch.max(torch.min(mat_max, self.material), mat_min)

            # broadcast stiffness params to all tets
            self.model.tet_materials = mat_val.expand((self.model.tet_count, 3)).contiguous()

            # forward dynamics
            with df.ScopedTimer("simulate", False):
                self.state = self.integrator.forward(self.model, self.state, self.sim_dt)
                self.sim_time += self.sim_dt

            # render
            with df.ScopedTimer("render", False):
                if (self.stage and render and (i % self.sim_substeps == 0)):
                    self.render_time += self.sim_dt * self.sim_substeps
                    self.renderer.update(self.state, self.render_time)

            # loss
            with df.ScopedTimer("loss", False):
                com_loss = torch.mean(self.state.particle_q, 0)

                # minimize y
                loss = loss - torch.norm(com_loss[1] - self.target)

        return loss

    def run(self):

        with torch.no_grad():
            l = self.loss()

        if (self.stage):
            self.stage.Save()

    def train(self, mode='gd'):

        # param to train
        self.step_count = 0
        render_freq = 1

        optimizer = None
        params = [
            self.material,
        ]

        def closure():

            if optimizer:
                optimizer.zero_grad()

            # render every N steps
            render = False
            if ((self.step_count % render_freq) == 0):
                render = True

            # with torch.autograd.detect_anomaly():
            with df.ScopedTimer("forward"):
                l = self.loss(render)

            with df.ScopedTimer("backward"):
                l.backward()

            print(self.material)
            print(self.material.grad)

            print(str(self.step_count) + ": " + str(l))
            self.step_count += 1

            with df.ScopedTimer("save"):
                try:
                    if (render):
                        self.stage.Save()
                except:
                    print("USD save error")

            return l

        with df.ScopedTimer("step"):

            if (mode == 'gd'):

                # simple Gradient Descent
                for i in range(self.train_iters):

                    closure()

                    with torch.no_grad():
                        for param in params:
                            param -= self.train_rate * param.grad
            else:

                # L-BFGS
                if (mode == 'lbfgs'):
                    optimizer = torch.optim.LBFGS(params, lr=1.0, tolerance_grad=1.e-5, tolerance_change=0.01, line_search_fn="strong_wolfe")

                # Adam
                if (mode == 'adam'):
                    optimizer = torch.optim.Adam(params, lr=self.train_rate)

                # SGD
                if (mode == 'sgd'):
                    optimizer = torch.optim.SGD(params, lr=self.train_rate, momentum=0.5, nesterov=True)

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

beam = Beam(device='cpu')
#beam.run()

#beam.train('lbfgs')
beam.train('gd')
