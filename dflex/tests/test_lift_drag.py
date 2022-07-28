# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import torch
from torch.utils.tensorboard import SummaryWriter

# include parent path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

from pxr import Usd, UsdGeom, Gf

# uncomment to output timers
df.ScopedTimer.enabled = False


class Cloth:

    sim_duration = 2.0       # seconds
    sim_substeps = 16
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    render_time = 0.0

    train_iters = 4
    train_rate = 0.01 / sim_substeps

    phase_count = 4

    def __init__(self, adapter='cpu'):

        torch.manual_seed(42)

        height = 2.5

        builder = df.sim.ModelBuilder()
        builder.add_cloth_grid(pos=(0.0, height, 0.0),
                               rot=df.quat_from_axis_angle((1.0, 0.5, 0.0), math.pi * 0.5),
                               vel=(0.0, 0.0, 0.0),
                               dim_x=16,
                               dim_y=16,
                               cell_x=0.125,
                               cell_y=0.125,
                               mass=1.0)                                                    #, fix_left=True, fix_right=True, fix_top=True, fix_bottom=True)

        self.model = builder.finalize(adapter)
        self.model.tri_ke = 10000.0
        self.model.tri_ka = 10000.0
        self.model.tri_kd = 100.0
        self.model.tri_lift = 10.0
        self.model.tri_drag = 5.0

        self.model.contact_ke = 1.e+4
        self.model.contact_kd = 1000.0
        self.model.contact_kf = 1000.0
        self.model.contact_mu = 0.5

        self.model.particle_radius = 0.01
        self.model.ground = False

        self.target = torch.tensor((8.0, 0.0, 0.0), device=adapter)
        self.initial_velocity = torch.tensor((1.0, 0.0, 0.0), requires_grad=True, device=adapter)

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/drag.usd")

        self.renderer = df.render.UsdRenderer(self.model, self.stage)
        self.renderer.draw_points = True
        self.renderer.draw_springs = True
        self.renderer.draw_shapes = True

        self.renderer.add_sphere(self.target.tolist(), 0.1, "target")

        self.integrator = df.sim.SemiImplicitIntegrator()

    def loss(self, render=True):

        # reset state
        self.sim_time = 0.0
        self.state = self.model.state()

        self.state.particle_qd = self.state.particle_qd + self.initial_velocity

        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        # run simulation
        for i in range(0, self.sim_steps):

            with df.ScopedTimer("simulate", False):
                self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

            with df.ScopedTimer("render", False):
                if (render and (i % self.sim_substeps == 0)):
                    self.render_time += self.sim_dt * self.sim_substeps
                    self.renderer.update(self.state, self.render_time)

            self.sim_time += self.sim_dt

            # compute loss
            with df.ScopedTimer("loss", False):
                com_pos = torch.mean(self.state.particle_q, 0)
                com_vel = torch.mean(self.state.particle_qd, 0)

                # use integral of velocity over course of the run
                loss = loss + torch.norm(com_pos - self.target)

        return loss

    def run(self):

        l = self.loss()
        self.stage.Save()

    def train(self, mode='gd'):

        writer = SummaryWriter()
        writer.add_hparams({"lr": self.train_rate, "mode": mode}, {})

        # param to train
        self.step_count = 0
        self.render_steps = 1

        optimizer = None
        param = self.initial_velocity

        def closure():

            if (optimizer):
                optimizer.zero_grad()

            # render every N steps
            render = False
            if ((self.step_count % self.render_steps) == 0):
                render = True

            with df.ScopedTimer("forward"):
                l = self.loss(render)

            with df.ScopedTimer("backward"):
                l.backward()

            with df.ScopedTimer("save"):
                if (render):
                    self.stage.Save()

            print(str(self.step_count) + ": " + str(l))
            writer.add_scalar("loss", l.item(), self.step_count)
            writer.flush()

            self.step_count += 1

            return l

        with df.ScopedTimer("step"):

            if (mode == 'gd'):

                # simple Gradient Descent
                for i in range(self.train_iters):

                    closure()

                    with torch.no_grad():
                        param -= self.train_rate * param.grad
                        param.grad.zero_()
            else:

                # L-BFGS
                if (mode == 'lbfgs'):
                    optimizer = torch.optim.LBFGS([param], lr=0.1, tolerance_grad=1.e-5, tolerance_change=0.01, line_search_fn="strong_wolfe")

                # Adam
                if (mode == 'adam'):
                    optimizer = torch.optim.Adam([param], lr=self.train_rate * 4.0)

                # SGD
                if (mode == 'sgd'):
                    optimizer = torch.optim.SGD([param], lr=self.train_rate * (1.0 / 32.0), momentum=0.8)

                # train
                for i in range(self.train_iters):
                    optimizer.step(closure)

        writer.close()

    def save(self, file):
        torch.save(self.network, file)

    def load(self, file):
        self.network = torch.load(file)
        self.network.eval()


#---------

cloth = Cloth(adapter='cpu')
cloth.train('lbfgs')
#cloth.run()
