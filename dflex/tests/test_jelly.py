# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import torch
import time
import timeit
import cProfile

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

from pxr import Usd, UsdGeom, Gf


class Bending:

    sim_duration = 5.0       # seconds
    sim_substeps = 16
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 200
    train_rate = 0.01        

    phase_count = 8
    phase_step = math.pi / phase_count * 2.0
    phase_freq = 2.5

    def __init__(self, adapter='cpu'):

        torch.manual_seed(42)

        r = df.quat_multiply(df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0), df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5))

        builder = df.sim.ModelBuilder()
      
        mesh = Usd.Stage.Open("assets/jellyfish.usda")
        geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/Icosphere/Icosphere"))

        points = geom.GetPointsAttr().Get()
        indices = geom.GetFaceVertexIndicesAttr().Get()
        counts = geom.GetFaceVertexCountsAttr().Get()

        face_materials = [-1] * len(counts)
        face_subsets = UsdGeom.Subset.GetAllGeomSubsets(geom)

        for i, s in enumerate(face_subsets):
            face_subset_indices = s.GetIndicesAttr().Get()

            for f in face_subset_indices:
                face_materials[f] = i

        active_material = 0
        active_scale = []

        def add_edge(f0, f1):
            if (face_materials[f0] == active_material and face_materials[f1] == active_material):
                active_scale.append(1.0)
            else:
                active_scale.append(0.0)

        builder.add_cloth_mesh(pos=(0.0, 2.5, 0.0),
                               rot=r,
                               scale=1.0,
                               vel=(0.0, 0.0, 0.0),
                               vertices=points,
                               indices=indices,
                               edge_callback=add_edge,
                               density=100.0)

        self.model = builder.finalize(adapter)
        self.model.tri_ke = 5000.0
        self.model.tri_ka = 5000.0
        self.model.tri_kd = 100.0
        self.model.tri_lift = 1000.0
        self.model.tri_drag = 0.0

        self.model.edge_ke = 20.0
        self.model.edge_kd = 1.0       #2.5

        self.model.contact_ke = 1.e+4
        self.model.contact_kd = 0.0
        self.model.contact_kf = 1000.0
        self.model.contact_mu = 0.5

        self.model.particle_radius = 0.01
        self.model.ground = False
        self.model.gravity = torch.tensor((0.0, 0.0, 0.0), device=adapter)

        # training params
        self.target_pos = torch.tensor((4.0, 2.0, 0.0), device=adapter)

        self.rest_angle = self.model.edge_rest_angle

        # one fully connected layer + tanh activation
        self.network = torch.nn.Sequential(torch.nn.Linear(self.phase_count, self.model.edge_count, bias=False), torch.nn.Tanh()).to(adapter)

        self.activation_strength = math.pi * 0.3
        self.activation_scale = torch.tensor(active_scale, device=adapter)
        self.activation_penalty = 0.0

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/jelly.usd")

        if (self.stage):
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

        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        for i in range(0, self.sim_steps):

            # build sinusoidal input phases
            phases = torch.zeros(self.phase_count, device=self.model.adapter)
            for p in range(self.phase_count):
                phases[p] = math.sin(self.phase_freq*self.sim_time + p * self.phase_step)

            # compute activations (rest angles)
            activation = (self.network(phases)) * self.activation_strength * self.activation_scale
            self.model.edge_rest_angle = self.rest_angle + activation

            # forward dynamics
            with df.ScopedTimer("simulate", False):
                self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

            # render
            with df.ScopedTimer("render", False):
                if (self.stage and render and (i % self.sim_substeps == 0)):
                    self.sim_time += self.sim_dt * self.sim_substeps
                    self.renderer.update(self.state, self.sim_time)

            # loss
            with df.ScopedTimer("loss", False):
                com_loss = torch.mean(self.state.particle_qd * self.model.particle_mass[:, None], 0)
                act_loss = torch.norm(activation) * self.activation_penalty

                loss = loss - com_loss[1] - act_loss

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

        def closure():

            optimizer.zero_grad()

            # render every N steps
            render = False
            if ((self.step_count % render_freq) == 0):
                render = True

            with df.ScopedTimer("forward"):
                l = self.loss(render)

            with df.ScopedTimer("backward"):
                l.backward()

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
                    optimizer = torch.optim.SGD(self.network.parameters(), lr=self.train_rate, momentum=0.8, nesterov=True)

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
#bending.load('jelly_10358.net')
#bending.run()

#bending.train('lbfgs')
bending.train('sgd')
