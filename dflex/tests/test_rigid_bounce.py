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

class RigidBounce:

    frame_dt = 1.0/60.0

    episode_duration = 2.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 16
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
    
    sim_time = 0.0

    train_iters = 128
    train_rate = 0.01

    ground = True

    name = "rigid_bounce"

    def __init__(self, depth=1, mode='numpy', render=True, adapter='cpu'):

        torch.manual_seed(42)

        builder = df.sim.ModelBuilder()

        self.adapter = adapter
        self.mode = mode
        self.render = render


        builder.add_articulation()

        # add sphere
        link = builder.add_link(-1, df.transform((0.0, 0.0, 0.0), df.quat_identity()), (0,0,0), df.JOINT_FREE)
        shape = builder.add_shape_sphere(
                link, 
                (0.0, 0.0, 0.0), 
                df.quat_identity(), 
                radius=0.1,
                ke=1.e+4,
                kd=10.0,
                kf=1.e+2,
                mu=0.25)

        builder.joint_q[1] = 1.0

        #v_s = df.get_body_twist((0.0, 0.0, 0.0), (1.0, -1.0, 0.0), builder.joint_q[0:3])
        
        w_m = (0.0, 0.0, 3.0)           # angular velocity (expressed in world space)
        v_m = (0.0, 0.0, 0.0)           # linear velocity at center of mass (expressed in world space)
        p_m = builder.joint_q[0:3]      # position of the center of mass (expressed in world space)
        
        # set body0 twist
        builder.joint_qd[0:6] = df.get_body_twist(w_m, v_m, p_m)

        # get decomposed velocities
        print(df.get_body_angular_velocity(builder.joint_qd[0:6]))
        print(df.get_body_linear_velocity(builder.joint_qd[0:6], p_m))


        # finalize model
        self.model = builder.finalize(adapter)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=adapter)
        

        # initial velocity

        #self.model.joint_qd[3] = 0.5
        #self.model.joint_qd[4] = -0.5
        #self.model.joint_qd[2] = 1.0

        self.model.joint_qd.requires_grad_()

        self.target = torch.tensor((1.0, 1.0, 0.0), dtype=torch.float32, device=adapter)

        #-----------------------
        # set up Usd renderer
        if (self.render):
            
            self.stage = Usd.Stage.CreateNew("outputs/" + self.name + ".usd")

            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

            self.renderer.add_sphere(self.target.tolist(), 0.1, "target")

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

        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        for f in range(0, self.episode_frames):
                
            # df.config.no_grad = True
            #df.config.verify_fp = True

            # simulate
            with df.ScopedTimer("fk-id-dflex", detailed=False, active=False):

                for i in range(0, self.sim_substeps):
                    
                    self.state = self.integrator.forward(self.model, self.state, self.sim_dt)
                    self.sim_time += self.sim_dt

            # render
            with df.ScopedTimer("render", False):

                if (self.render):
                    self.render_time += self.frame_dt
                    self.renderer.update(self.state, self.render_time)
                    try:
                        self.stage.Save()
                    except:
                        print("USD save error")

        #loss = loss + torch.dot(self.state.joint_qd[3:6], self.state.joint_qd[3:6])*self.balance_penalty*discount
        pos = self.state.joint_q[0:3]
        
        loss = torch.norm(pos-self.target)
        return loss

    def run(self):
        
        df.config.no_grad = True
    
        #with torch.no_grad():
        l = self.loss()

    def verify(self, eps=1.e-4):
       
        frame = 60

        params = self.model.joint_qd
        n = len(params)

        # evaluate analytic gradient
        l = self.loss(render=False)
        l.backward()

        # evaluate numeric gradient
        grad_analytic = self.model.joint_qd.grad.tolist()
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

        params = [self.model.joint_qd]

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
        torch.save(self.model.joint_qd, "outputs/" + self.name + ".pt")

    def load(self):
        self.model.joint_qd = torch.load("outputs/" + self.name + ".pt")


#---------

robot = RigidBounce(depth=1, mode='dflex', render=True, adapter='cpu')

#df.config.check_grad = True
#df.config.no_grad = True
robot.run()

#df.config.verify_fp = True

#robot.load()
#robot.train(mode='lbfgs')

#robot.verify(eps=1.e-3)
