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


class Bending:

    sim_duration = 10.0      # seconds
    sim_substeps = 32
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 200
    train_rate = 0.01

    def __init__(self, adapter='cpu'):

        torch.manual_seed(42)

        builder = df.sim.ModelBuilder()

        if (True):

            mesh = Usd.Stage.Open("assets/icosphere_open.usda")
            geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/Shell/Mesh"))

            #mesh = Usd.Stage.Open("assets/cylinder_long_open.usda")
            #geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/CylinderLong/CylinderLong"))

            points = geom.GetPointsAttr().Get()
            indices = geom.GetFaceVertexIndicesAttr().Get()
            counts = geom.GetFaceVertexCountsAttr().Get()

            linear_vel = np.array((1.0, 0.0, 0.0))
            angular_vel = np.array((0.0, 0.0, 0.0))
            center = np.array((0.0, 1.6, 0.0))
            radius = 0.5

            r = df.quat_multiply(df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0), df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5))

            builder.add_cloth_mesh(pos=center, rot=(0.0, 0.0, 0.0, 1.0), scale=radius, vel=(0.0, 0.0, 0.0), vertices=points, indices=indices, density=10.0)

            for i in range(len(builder.particle_qd)):

                v = np.cross(np.array(builder.particle_q) - center, angular_vel)
                builder.particle_qd[i] = v + linear_vel

            self.model = builder.finalize(adapter)
            self.model.tri_ke = 2000.0
            self.model.tri_ka = 2000.0
            self.model.tri_kd = 3.0
            self.model.tri_lift = 0.0
            self.model.tri_drag = 0.0

            self.model.edge_ke = 20.0
            self.model.edge_kd = 0.3
            self.model.gravity = torch.tensor((0.0, -10.0, 0.0), device=adapter)

        else:

            builder.add_particle(pos=(1.0, 2.0, 1.0), vel=(0.0, 0.0, 0.0), mass=0.0)
            builder.add_particle(pos=(1.0, 2.0, -1.0), vel=(0.0, 0.0, 0.0), mass=0.0)
            builder.add_particle(pos=(-1.0, 2.0, -1.0), vel=(0.0, 0.0, 0.0), mass=0.0)
            builder.add_particle(pos=(-1.0, 2.0, 1.0), vel=(0.0, 0.0, 0.0), mass=1.0)

            builder.add_triangle(0, 1, 2)
            builder.add_triangle(0, 2, 3)

            builder.add_edge(1, 3, 2, 0)
            builder.edge_rest_angle[0] = -math.pi * 0.6

            self.model = builder.finalize(adapter)
            self.model.tri_ke = 2000.0
            self.model.tri_ka = 2000.0
            self.model.tri_kd = 3.0
            self.model.tri_lift = 0.0
            self.model.tri_drag = 0.0

            self.model.edge_ke = 20.0
            self.model.edge_kd = 1.7

            self.model.gravity = torch.tensor((0.0, 0.0, 0.0), device=adapter)

        # contact params
        self.model.contact_ke = 1.e+4
        self.model.contact_kd = 1.0
        self.model.contact_kf = 1000.0
        self.model.contact_mu = 5.0
        self.model.particle_radius = 0.01
        self.model.ground = True

        # training params
        self.target_pos = torch.tensor((4.0, 2.0, 0.0), device=adapter)

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/bending.usd")

        self.renderer = df.render.UsdRenderer(self.model, self.stage)
        self.renderer.draw_points = True
        self.renderer.draw_springs = True
        self.renderer.draw_shapes = True
        self.renderer.add_sphere(self.target_pos.tolist(), 0.1, "target")

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

            # loss
            #com_loss = torch.mean(self.state.particle_qd*self.model.particle_mass[:, None], 0)
            #act_loss = torch.norm(activation)*self.activation_penalty

            #loss = loss - com_loss[1]

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

bending = Bending(adapter='cpu')
bending.run()

#bending.train('lbfgs')
#bending.train('sgd')
