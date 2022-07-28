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

# uncomment to output timers
df.ScopedTimer.enabled = True


class Cloth:

    sim_duration = 10.0                          # seconds
    sim_substeps = 16
    sim_dt = (1.0 / 60.0) / sim_substeps         # 60 frames per second, 16 updates between frames,
                                                 # sim_steps = int(sim_duration/sim_dt)
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    render_time = 0.0

    train_iters = 100
    train_rate = 0.01

    phase_count = 4

    def __init__(self, dim=20, mode="cloth", adapter='cpu'):

        torch.manual_seed(42)

        height = 4.0

        builder = df.sim.ModelBuilder()

        # builder.add_particle(pos=(2.5, 3.0, 2.5), vel=(0.0, 0.0, 0.0), mass=1.0)
        # builder.add_particle(pos=(2.5, 4.0, 2.5), vel=(0.0, 0.0, 0.0), mass=1.0)
        # builder.add_particle(pos=(2.5, 5.0, 2.5), vel=(0.0, 0.0, 0.0), mass=1.0)

        builder.add_cloth_grid(pos=(0.0, height, 0.0),
                               rot=df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi / 2),
                               vel=(0, 5.0, 0.0),
                               dim_x=dim,
                               dim_y=dim,
                               cell_x=0.2,
                               cell_y=0.2,
                               mass=400 / (dim**2))                                       #, fix_left=True, fix_right=True, fix_top=True, fix_bottom=True)

        usd = Usd.Stage.Open("assets/box.usda")
        geom = UsdGeom.Mesh(usd.GetPrimAtPath("/Cube/Cube"))

        points = geom.GetPointsAttr().Get()
        indices = geom.GetFaceVertexIndicesAttr().Get()
        counts = geom.GetFaceVertexCountsAttr().Get()

        mesh = df.sim.Mesh(points, indices)

        rigid = builder.add_rigid_body(pos=(2.5, 3.0, 2.5),
                                       rot=df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0),
                                       vel=(0.0, 0.0, 0.0),
                                       omega=(0.0, 0.0, 0.0))
        shape = builder.add_shape_mesh(rigid, mesh=mesh, scale=(0.5, 0.5, 0.5), density=100.0, ke=1.e+5, kd=1000.0, kf=1000.0, mu=0.5)

        # rigid = builder.add_rigid_body(pos=(2.5, 5.0, 2.5), rot=df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi*0.0), vel=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0))
        # shape = builder.add_shape_mesh(rigid, mesh=mesh, scale=(0.5, 0.5, 0.5), density=100.0, ke=1.e+5, kd=1000.0, kf=1000.0, mu=0.5)

        # attach0 = 1
        # attach1 = 21
        # attach2 = 423
        # attach3 = 443

        # anchor0 = builder.add_particle(pos=builder.particle_x[attach0]-(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.0)
        # anchor1 = builder.add_particle(pos=builder.particle_x[attach1]+(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.0)
        # anchor2 = builder.add_particle(pos=builder.particle_x[attach2]-(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.0)
        # anchor3 = builder.add_particle(pos=builder.particle_x[attach3]+(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.0)

        # builder.add_spring(anchor0, attach0, 500.0, 1000.0, 0)
        # builder.add_spring(anchor1, attach1, 10000.0, 1000.0, 0)
        # builder.add_spring(anchor2, attach2, 10000.0, 1000.0, 0)
        # builder.add_spring(anchor3, attach3, 10000.0, 1000.0, 0)

        self.model = builder.finalize(adapter)

        # self.model.tri_ke = 10000.0
        # self.model.tri_ka = 10000.0
        # self.model.tri_kd = 100.0

        self.model.tri_ke = 5000.0
        self.model.tri_ka = 5000.0
        self.model.tri_kd = 100.0
        self.model.tri_lift = 50.0
        self.model.tri_drag = 0.0

        self.model.contact_ke = 1.e+4
        self.model.contact_kd = 1000.0
        self.model.contact_kf = 1000.0
        self.model.contact_mu = 0.5

        self.model.particle_radius = 0.1
        self.model.ground = True
        self.model.tri_collisions = True

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/cloth_collision.usd")

        self.renderer = df.render.UsdRenderer(self.model, self.stage)
        self.renderer.draw_points = True
        self.renderer.draw_springs = True
        self.renderer.draw_shapes = True

        self.integrator = df.sim.SemiImplicitIntegrator()

    def loss(self, render=True):

        # reset state
        self.sim_time = 0.0
        self.state = self.model.state()
        self.model.collide(self.state)

        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        with df.ScopedTimer("forward", False):
            # run simulation
            for i in range(0, self.sim_steps):
                with df.ScopedTimer("simulate", False):
                    self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

                with df.ScopedTimer("render", False):
                    if (render and (i % self.sim_substeps == 0)):
                        self.render_time += self.sim_dt * self.sim_substeps
                        self.renderer.update(self.state, self.render_time)

                if (self.state.particle_q != self.state.particle_q).sum() != 0:
                    print("NaN found in state")
                    import pdb
                    pdb.set_trace()

                self.sim_time += self.sim_dt

        # compute loss
        with df.ScopedTimer("loss", False):
            com_pos = torch.mean(self.state.particle_q, 0)
            com_vel = torch.mean(self.state.particle_qd, 0)

            # use integral of velocity over course of the run
            loss = loss - com_pos[1]
            return loss

    def run(self):

        l = self.loss()
        self.stage.Save()

    def train(self, mode='gd'):

        # param to train
        self.step_count = 0
        self.render_steps = 1

        optimizer = None

        def closure():
            # render every N steps
            render = False
            if ((self.step_count % self.render_steps) == 0):
                render = True

            # with torch.autograd.detect_anomaly():
            with df.ScopedTimer("forward", False):
                l = self.loss(render)

            with df.ScopedTimer("backward", False):
                l.backward()

            with df.ScopedTimer("save", False):
                if (render):
                    self.stage.Save()

            print(str(self.step_count) + ": " + str(l))
            self.step_count += 1

            return l

        with df.ScopedTimer("step"):
            if (mode == 'gd'):
                param = self.model.spring_rest_length

                # simple Gradient Descent
                for i in range(self.train_iters):
                    closure()

                    with torch.no_grad():
                        param -= self.train_rate * param.grad

                    param.grad.data.zero_()
            else:

                # L-BFGS
                if (mode == 'lbfgs'):
                    optimizer = torch.optim.LBFGS([self.model.spring_rest_length],
                                                  lr=0.01,
                                                  tolerance_grad=1.e-5,
                                                  tolerance_change=0.01,
                                                  line_search_fn="strong_wolfe")

                # Adam
                if (mode == 'adam'):
                    optimizer = torch.optim.Adam([self.model.spring_rest_length], lr=self.train_rate * 4.0)

                # SGD
                if (mode == 'sgd'):
                    optimizer = torch.optim.SGD([self.model.spring_rest_length], lr=self.train_rate * 0.01, momentum=0.8)

                # train
                for i in range(self.train_iters):
                    optimizer.zero_grad()
                    optimizer.step(closure)

    def save(self, file):
        torch.save(self.network, file)

    def load(self, file):
        self.network = torch.load(file)
        self.network.eval()


#---------

cloth = Cloth(adapter='cuda')
cloth.run()
         # cloth.train('adam')

         # for dim in range(20, 400, 20):
         #     cloth = Cloth(dim=dim)
         #     cloth.run()
