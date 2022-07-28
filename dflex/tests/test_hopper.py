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

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

from pxr import Usd, UsdGeom, Gf

import test_util

class Robot:

    frame_dt = 1.0/60.0

    episode_duration = 2.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 16
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
    
    sim_time = 0.0

    train_iters = 1024
    train_rate = 0.001

    ground = True

    name = "hopper"

    regularization = 1.e-3

    def __init__(self, depth=1, mode='numpy', render=True, adapter='cpu'):

        torch.manual_seed(42)

        builder = df.sim.ModelBuilder()

        self.adapter = adapter
        self.mode = mode
        self.render = render   

        # humanoid
        test_util.urdf_load(
            builder, 
            "assets/hopper.urdf", 
            #df.transform((0.0, 1.35, 0.0), df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)), 
            df.transform((0.0, 0.65, 0.0), df.quat_identity()), 
            floating=True,
            shape_ke=1.e+3*2.0,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.5)
        
        # set pd-stiffness
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 10.0
            builder.joint_target_kd[i] = 1.0

        # finalize model
        self.model = builder.finalize(adapter)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=adapter)
        
        self.model.joint_q.requires_grad_()
        self.model.joint_qd.requires_grad_()

        self.actions = torch.zeros((self.episode_frames, len(self.model.joint_qd)), dtype=torch.float32, device=adapter, requires_grad=True)

        self.action_strength = 20.0
        self.action_penalty = 0.01

        self.balance_reward = 15.0
        self.forward_reward = 1.0

        self.discount_scale = 3.0
        self.discount_factor = 0.5

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

    def loss(self, render=True):

        #---------------
        # run simulation

        self.sim_time = 0.0
        self.state = self.model.state()

        if (self.model.ground):
            self.model.collide(self.state)

        loss = torch.zeros(1, requires_grad=True, dtype=torch.float32, device=self.model.adapter)

        for f in range(0, self.episode_frames):
                
            # df.config.no_grad = True
            #df.config.verify_fp = True

            # simulate
            with df.ScopedTimer("fk-id-dflex", detailed=False, active=False):

                for i in range(0, self.sim_substeps):
                    
                    self.state.joint_act = self.actions[f]

                    self.state = self.integrator.forward(self.model, self.state, self.sim_dt)
                    self.sim_time += self.sim_dt

                    # discount_time = self.sim_time
                    # discount = math.pow(self.discount_factor, discount_time*self.discount_scale)

                loss = loss - self.state.joint_q[1]*self.state.joint_q[1]*self.balance_reward


            # render
            with df.ScopedTimer("render", False):

                if (self.render):
                    self.render_time += self.frame_dt
                    self.renderer.update(self.state, self.render_time)
        
        if (self.render):
            try:
                self.stage.Save()
            except:
                print("USD save error")
 
        return loss

    def run(self):
        
        df.config.no_grad = True
    
        #with torch.no_grad():
        l = self.loss()

    def verify(self, eps=1.e-4):
       
        frame = 60

        params = self.actions[frame]
        n = len(params)

        # evaluate analytic gradient
        l = self.loss(render=False)
        l.backward()

        # evaluate numeric gradient
        grad_analytic = self.actions.grad[frame].numpy()
        grad_numeric = np.zeros(n)

        with torch.no_grad():
            
            df.config.no_grad = True

            for i in range(n):
                mid = params[i].item()

                params[i] = mid - eps
                left = self.loss(render=False)
                
                params[i] = mid + eps
                right = self.loss(render=False)

                # reset
                params[i] = mid

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

            #print("vel: " + str(params[0]))
            #print("grad: " + str(params[0].grad))
            #print("--------")

            print(str(self.step_count) + ": " + str(l))
            self.step_count += 1

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

#df.config.check_grad = True
#df.config.no_grad = True
#robot.run()

#df.config.verify_fp = True

#robot.load()
robot.train(mode='lbfgs')

#robot.verify(eps=1.e-3)
