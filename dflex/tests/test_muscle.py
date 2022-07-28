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

# to allow tests to import the module they belong to
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

from pxr import Usd, UsdGeom, Gf

import test_util

class MuscleTest:

    sim_duration = 4.0      # seconds
    sim_substeps = 16
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    
    sim_time = 0.0

    train_iters = 128
    train_rate = 10.0

    ground = False

    name = "muscle"

    regularization = 1.e-3
    
    def __init__(self, depth=1, mode='numpy', render=True, adapter='cpu'):

        torch.manual_seed(42)

        builder = df.sim.ModelBuilder()

        self.adapter = adapter
        self.mode = mode
        self.render = render

        length = 0.5
        width = 0.1
        radius = 0.05

        builder.add_articulation()

        body = -1
        for i in range(2):
            if (i == 0):
                body = builder.add_link(body, df.transform((2.0*length, 0.0, 0.0), df.quat_identity()), axis=(0.0, 0.0, 1.0), damping=1.0, stiffness=0.0, type=df.JOINT_FIXED)
            else:
                body = builder.add_link(body, df.transform((2.0*length, 0.0, 0.0), df.quat_identity()), axis=(0.0, 0.0, 1.0), damping=1.0, stiffness=0.0, type=df.JOINT_BALL)

            shape = builder.add_shape_box(body, pos=(length, 0.0, 0.0), hx=length, hy=width, hz=width)

        builder.add_muscle([0, 1], [(length*2.0 - 0.25, width + 0.05, 0.0), (0.25, width + 0.05, 0.0)], 1.0, 1.0, 1.0, 1.0, 1.0)
        builder.add_muscle([0, 1], [(length*2.0 - 0.25, -width - 0.05, 0.0), (0.25, -width - 0.05, 0.0)], 1.0, 1.0, 1.0, 1.0, 1.0)
        builder.muscle_activation[0] = 1000.0
        builder.muscle_activation[1] = 0.0

        self.pole_angle_penalty = 10.0
        self.pole_velocity_penalty = 0.5

        self.cart_action_penalty = 1.e-7
        self.cart_velocity_penalty = 1.0
        self.cart_position_penalty = 2.0

        # finalize model
        self.model = builder.finalize(adapter)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), device=adapter)
        #self.model.gravity = torch.tensor((0.0, 0.0, 0.0), device=adapter)

        self.activations = torch.zeros((self.sim_steps, 2), dtype=torch.float32, device=adapter, requires_grad=True)
        self.model.joint_q.requires_grad = True
        self.model.joint_qd.requires_grad = True
        self.model.muscle_activation.requires_grad = True

        #-----------------------
        # set up Usd renderer
        if (self.render):
            
            self.stage = Usd.Stage.CreateNew("outputs/" + self.name + ".usd")

            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

        self.integrator = df.sim.SemiImplicitIntegrator()

    def set_target(self, x, name):

        self.target = torch.tensor(x, device='cpu')

        self.renderer.add_sphere(self.target.tolist(), 0.1, name)

    def loss(self):

        #---------------
        # run simulation

        self.sim_time = 0.0
        
        # initial state
        self.state = self.model.state() 

        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        #self.model.muscle_activation = torch.zeros_like(self.model.muscle_activation)

        for i in range(0, self.sim_steps):

            # apply actions
            #for m in range(self.model.muscle_count):
            #self.model.muscle_activation = self.activations[i]

            # simulate
            with df.ScopedTimer("fd", active=False):
                self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

            # render
            with df.ScopedTimer("render", active=False):
                if (self.render and (i % self.sim_substeps == 0)):

                    with torch.no_grad():

                     for m in range(self.model.muscle_count):

                        start = self.model.muscle_start[m]
                        end = self.model.muscle_start[m+1]

                        points = []

                        for w in range(start, end):
                            
                            link = self.model.muscle_links[w]
                            point = self.model.muscle_points[w]

                            X_sc = df.transform_expand(self.state.body_X_sc[link].tolist())

                            points.append(Gf.Vec3f(df.transform_point(X_sc, point).tolist()))
                            
                        self.renderer.add_line_strip(points, name=("muscle_0" + str(m)), radius=0.02, color=(self.activations[i][m]/1000.0 + 0.5, 0.2, 0.5), time=self.render_time)

                        # render scene
                        self.render_time += self.sim_dt * self.sim_substeps
                        self.renderer.update(self.state, self.render_time)

            self.sim_time += self.sim_dt

            loss = loss + self.state.joint_q[2]*self.state.joint_q[2]

        return loss

    def run(self):

        l = self.loss()

        if (self.render):
            self.stage.Save()

    def verify(self, eps=1.e-4):
       
        params = self.actions
        n = 1#len(params)

        self.render = False

        # evaluate analytic gradient
        l = self.loss()
        l.backward()

        # evaluate numeric gradient
        grad_analytic = params.grad.cpu().numpy()
        grad_numeric = np.zeros(n)

        with torch.no_grad():
            
            df.config.no_grad = True

            for i in range(1):
                mid = params[0][i].item()

                params[0][i] = mid - eps
                left = self.loss()
                
                params[0][i] = mid + eps
                right = self.loss()

                # reset
                params[0][i] = mid

                # numeric grad
                grad_numeric[i] = (right-left)/(2.0*eps)

        # report
        print("grad_numeric: " + str(grad_numeric))
        print("grad_analytic: " + str(grad_analytic))

 
    def train(self, mode='gd'):

        # param to train
        self.step_count = 0
        self.best_loss = math.inf

        render_freq = 1
        
        optimizer = None

        params = [self.activations]

        def closure():

            if (optimizer):
                optimizer.zero_grad()

            # render ever y N steps
            render = False
            if ((self.step_count % render_freq) == 0):
                render = True

            with df.ScopedTimer("forward"):
                #with torch.autograd.detect_anomaly():
                l = self.loss()

            with df.ScopedTimer("backward"):
                #with torch.autograd.detect_anomaly():
                l.backward()

            # for e in range(self.env_count):
            #     print(self.actions.grad[e][0:20])

            print(str(self.step_count) + ": " + str(l))
            self.step_count += 1

            with df.ScopedTimer("save"):
                try:
                    if (render):
                        self.stage.Save()
                except:
                    print("USD save error")

            # save best trajectory
            if (l.item() < self.best_loss):
                self.save()
                self.best_loss = l.item()

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
                    optimizer = torch.optim.LBFGS(params, lr=1.0, tolerance_grad=1.e-9, line_search_fn="strong_wolfe")

                # Adam
                if (mode == 'adam'):
                    optimizer = torch.optim.Adam(params, lr=self.train_rate)

                # SGD
                if (mode == 'sgd'):
                    optimizer = torch.optim.SGD(params, lr=self.train_rate, momentum=0.8, nesterov=True)

                # train
                for i in range(self.train_iters):
                    print("Step: " + str(i))
                    optimizer.step(closure)

                # final save
                try:
                    if (render):
                        self.stage.Save()
                except:
                    print("USD save error")

    def save(self):
        torch.save(self.activations, "outputs/" + self.name + ".pt")

    def load(self):
        self.activations = torch.load("outputs/" + self.name + ".pt")


#---------

test = MuscleTest(depth=1, mode='dflex', render=True, adapter='cpu')

#df.config.no_grad = True
#df.config.check_grad = True
#df.config.verify_fp = True

#test.load()
test.run()
#test.train('lbfgs')

