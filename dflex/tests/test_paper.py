# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import torch
import time
import cProfile

import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

from pxr import Usd, UsdGeom, Gf


class Paper:

    sim_duration = 10.0      # seconds
    sim_substeps = 64
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 200
    train_rate = 0.01

    def __init__(self, adapter='cpu'):

        torch.manual_seed(42)
        np.random.seed(42)

        builder = df.sim.ModelBuilder()

        mesh = Usd.Stage.Open("assets/paper.usda")
        geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/Grid/Grid"))

        # mesh = Usd.Stage.Open("assets/dart.usda")
        # geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/planes_001/planes_001"))

        points = geom.GetPointsAttr().Get()
        indices = geom.GetFaceVertexIndicesAttr().Get()
        counts = geom.GetFaceVertexCountsAttr().Get()

        center = np.array((0.0, 10.0, 0.0))
        radius = 5.0

        for i in range(1):

            center = np.array([0.0, 5.0, 0.0]) + np.random.ranf((3, )) * 10.0
            axis = df.normalize(np.random.ranf((3, )))
            angle = np.random.ranf(1, ) * math.pi

            builder.add_cloth_mesh(pos=center,
                                   rot=df.quat_from_axis_angle(axis, angle),
                                   scale=radius,
                                   vel=(0.0, 0.0, 0.0),
                                   vertices=points,
                                   indices=indices,
                                   density=100.0)

        self.model = builder.finalize(adapter)
        self.model.tri_ke = 2000.0
        self.model.tri_ka = 2000.0
        self.model.tri_kd = 100.0
        self.model.tri_lift = 50.0
        self.model.tri_drag = 0.5

        self.model.contact_ke = 1.e+4
        self.model.contact_kd = 1000.0
        self.model.contact_kf = 2000.0
        self.model.contact_mu = 0.5

        self.model.edge_ke = 20.0
        self.model.edge_kd = 0.3

        self.model.gravity = torch.tensor((0.0, -10.0, 0.0), device=adapter)
        self.model.particle_radius = 0.01
        self.model.ground = True

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/paper.usd")

        self.renderer = df.render.UsdRenderer(self.model, self.stage)
        self.renderer.draw_points = True
        self.renderer.draw_springs = True
        self.renderer.draw_shapes = True

        self.integrator = df.sim.SemiImplicitIntegrator()

    def loss(self, render=True):

        #-----------------------
        # run simulation

        self.state = self.model.state()

        loss = torch.zeros(1, requires_grad=True)

        for i in range(0, self.sim_steps):

            # forward dynamics
            self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

            # render
            if (render and (i % self.sim_substeps == 0)):
                self.sim_time += self.sim_dt * self.sim_substeps
                self.renderer.update(self.state, self.sim_time)

        return loss

    def run(self):

        with torch.no_grad():
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
                optimizer = torch.optim.LBFGS(self.network.parameters(), lr=1.0, tolerance_grad=1.e-5, tolerance_change=0.01, line_search_fn="strong_wolfe")

            # Adam
            if (mode == 'adam'):
                optimizer = torch.optim.Adam(self.network.parameters(), lr=self.train_rate)

            # SGD
            if (mode == 'sgd'):
                optimizer = torch.optim.SGD(self.network.parameters(), lr=self.train_rate, momentum=0.5)

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

paper = Paper(adapter='cpu')
paper.run()

#bending.train('lbfgs')
#bending.train('sgd')
