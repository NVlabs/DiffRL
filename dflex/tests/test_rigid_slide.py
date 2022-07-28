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
import tinyobjloader

import numpy as np

from pxr import Usd, UsdGeom, Gf


class RigidSlide:

    sim_duration = 3.0       # seconds
    sim_substeps = 16
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 64
    train_rate = 0.1

    discount_scale = 1.0
    discount_factor = 0.5

    def __init__(self, adapter='cpu'):

        torch.manual_seed(42)

        # load mesh
        usd = Usd.Stage.Open("assets/suzanne.usda")
        geom = UsdGeom.Mesh(usd.GetPrimAtPath("/Suzanne/Suzanne"))

        points = geom.GetPointsAttr().Get()
        indices = geom.GetFaceVertexIndicesAttr().Get()
        counts = geom.GetFaceVertexCountsAttr().Get()

        builder = df.sim.ModelBuilder()

        mesh = df.sim.Mesh(points, indices)

        articulation = builder.add_articulation()

        rigid = builder.add_link(
            parent=-1,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(0.0, 0.0, 0.0),
            type=df.JOINT_FREE)

        ke = 1.e+4
        kd = 1.e+3
        kf = 1.e+3
        mu = 0.5

        # shape = builder.add_shape_mesh(
        #     rigid, 
        #     mesh=mesh, 
        #     scale=(0.2, 0.2, 0.2), 
        #     density=1000.0, 
        #     ke=1.e+4, 
        #     kd=1000.0, 
        #     kf=1000.0, 
        #     mu=0.75)

        radius = 0.1

        #shape = builder.add_shape_sphere(rigid, pos=(0.0, 0.0, 0.0), ke=ke, kd=kd, kf=kf, mu=mu, radius=radius)
        #shape = builder.add_shape_capsule(rigid, pos=(0.0, 0.0, 0.0), radius=radius, half_width=0.5)
        shape = builder.add_shape_box(rigid, pos=(0.0, 0.0, 0.0), hx=radius, hy=radius, hz=radius, ke=ke, kd=kd, kf=kf, mu=mu)

        builder.joint_q[1] = radius

        self.model = builder.finalize(adapter)
        self.model.joint_qd.requires_grad = True

        self.vel = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=adapter, requires_grad=True)
        self.target = torch.tensor((3.0, 0.2, 0.0), device=adapter)

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/rigid_slide.usda")

        if (self.stage):
            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

            self.renderer.add_sphere(self.target.tolist(), 0.1, "target")

        self.integrator = df.sim.SemiImplicitIntegrator()

    def loss(self, render=True):

        #---------------
        # run simulation
       # construct contacts once at startup
        self.model.joint_qd = torch.cat((torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32, device=self.model.adapter), self.vel))

        self.sim_time = 0.0
        self.state = self.model.state()

        self.model.collide(self.state)

        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)
 
        for i in range(0, self.sim_steps):

            # forward dynamics
            with df.ScopedTimer("simulate", False):
                self.state = self.integrator.forward(self.model, self.state, self.sim_dt)
                self.sim_time += self.sim_dt

            # render
            with df.ScopedTimer("render", False):
                if (self.stage and render and (i % self.sim_substeps == 0)):
                    self.render_time += self.sim_dt * self.sim_substeps
                    self.renderer.update(self.state, self.render_time)

            #com = self.state.joint_q[0:3]
        com = self.state.body_X_sm[0, 0:3]

        loss = loss + torch.norm(com - self.target)
        return loss

    def run(self):

        #with torch.no_grad():
        l = self.loss()

        if (self.stage):
            self.stage.Save()

    def train(self, mode='gd'):

        # param to train
        self.step_count = 0
        render_freq = 1

        optimizer = None

        params = [self.vel]

        def closure():

            if (optimizer):
                optimizer.zero_grad()

            # render every N steps
            render = False
            if ((self.step_count % render_freq) == 0):
                render = True

            with df.ScopedTimer("forward"):
                #with torch.autograd.detect_anomaly():
                l = self.loss(render)

            with df.ScopedTimer("backward"):
                #with torch.autograd.detect_anomaly():
                l.backward()

            print("vel: " + str(params[0]))
            print("grad: " + str(params[0].grad))
            print("--------")

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
                        params[0] -= self.train_rate * params[0].grad
                        params[0].grad.zero_()
            else:

                # L-BFGS
                if (mode == 'lbfgs'):
                    optimizer = torch.optim.LBFGS(params, lr=1.0, tolerance_grad=1.e-5, tolerance_change=0.01, line_search_fn="strong_wolfe")

                # Adam
                if (mode == 'adam'):
                    optimizer = torch.optim.Adam(params, lr=self.train_rate)

                # SGD
                if (mode == 'sgd'):
                    optimizer = torch.optim.SGD(params, lr=self.train_rate, momentum=0.8, nesterov=True)

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

rigid = RigidSlide(adapter='cpu')
#rigid.run()
rigid.train('adam')
