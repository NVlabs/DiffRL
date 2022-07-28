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

class Robot:

    sim_duration = 4.0      # seconds
    sim_substeps = 4
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    
    sim_time = 0.0

    train_iters = 128
    train_rate = 10.0

    ground = False

    name = "franka"

    regularization = 1.e-3
    
    env_count = 1
    env_dofs = 2

    def __init__(self, depth=1, mode='numpy', render=True, adapter='cpu'):

        torch.manual_seed(42)

        builder = df.sim.ModelBuilder()

        self.adapter = adapter
        self.mode = mode
        self.render = render

        # cartpole
        for i in range(self.env_count):
            test_util.urdf_load(
                builder, 
                "assets/franka_description/robots/franka_panda.urdf", 
                df.transform((0.0, 0.0, 0.0), df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)), 
                floating=False,
                limit_ke=1.e+3,
                limit_kd=1.e+2)

        for i in range(len(builder.joint_target_kd)):
            builder.joint_target_kd[i] = 1.0

        # finalize model
        self.model = builder.finalize(adapter)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), device=adapter)
        #self.model.gravity = torch.tensor((0.0, 0.0, 0.0), device=adapter)
        
        self.model.joint_q.requires_grad_()
        self.model.joint_qd.requires_grad_()

        self.actions = torch.zeros((self.env_count, self.sim_steps), device=adapter, requires_grad=True)
        #self.actions = torch.zeros(1, device=adapter, requires_grad=True)

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

        if (self.render):
            traj = []
            for e in range(self.env_count):
                traj.append([])

        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        for i in range(0, self.sim_steps):

            # simulate
            with df.ScopedTimer("fd", detailed=False, active=False):
                self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

            # render
            with df.ScopedTimer("render", False):
                if (self.render and (i % self.sim_substeps == 0)):

                    with torch.no_grad():

                        # draw end effector tracer
                        # for e in range(self.env_count):
                        #     X_pole = df.transform_point(df.transform_expand(self.state.body_X_sc[e*3 + self.marker_body].tolist()), (0.0, 0.0, self.marker_offset))
                            
                        #     traj[e].append((X_pole[0], X_pole[1], X_pole[2]))
                            
                        #     # render trajectory
                        #     self.renderer.add_line_strip(traj[e], (1.0, 1.0, 1.0), self.render_time, "traj_" + str(e))

                        # render scene
                        self.render_time += self.sim_dt * self.sim_substeps
                        self.renderer.update(self.state, self.render_time)

            self.sim_time += self.sim_dt

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

        params = [self.actions]

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
        torch.save(self.actions, "outputs/" + self.name + ".pt")

    def load(self):
        self.actions = torch.load("outputs/" + self.name + ".pt")


#---------

robot = Robot(depth=1, mode='dflex', render=True, adapter='cpu')

#df.config.no_grad = True
#df.config.check_grad = True
#df.config.verify_fp = True

#robot.load()
robot.run()

#robot.train(mode='lbfgs')
#robot.verify(eps=1.e+1)
