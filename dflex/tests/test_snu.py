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


class HumanoidSNU:

    sim_duration = 1.0      # seconds
    sim_substeps = 16
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    
    sim_time = 0.0

    train_iters = 128
    train_rate = 0.01

    ground = False

    name = "humanoid_snu_neck"

    regularization = 1.e-3
    
    env_count = 16
    env_dofs = 2

    def __init__(self, depth=1, mode='numpy', render=True, adapter='cpu'):

        torch.manual_seed(41)

        builder = df.sim.ModelBuilder()

        self.adapter = adapter
        self.mode = mode
        self.render = render

        self.filter = {}

        if self.name == "humanoid_snu_arm":
            self.filter = { "ShoulderR", "ArmR", "ForeArmR", "HandR", "Torso", "Neck" }
            self.ground = False

        if self.name == "humanoid_snu_neck":
            self.filter = { "Torso", "Neck", "Head", "ShoulderR", "ShoulderL"}
            self.ground = False
        
        self.node_map, self.xform_map, self.mesh_map = test_util.parse_skeleton("assets/snu/arm.xml", builder, self.filter)
        self.muscles = test_util.parse_muscles("assets/snu/muscle284.xml", builder, self.node_map, self.xform_map)

        # set initial position 1m off the ground
        if self.name == "humanoid_snu":
            builder.joint_q[1] = 1.0

        # finalize model
        self.model = builder.finalize(adapter)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=adapter)
        #self.model.gravity = torch.tensor((0.0, 0.0, 0.0), device=adapter)
        
        self.activations = torch.zeros((1, len(self.muscles)), dtype=torch.float32, device=adapter, requires_grad=True)
        #self.activations = torch.rand((1, len(self.muscles)), dtype=torch.float32, device=adapter, requires_grad=True)

        self.model.joint_q.requires_grad = True
        self.model.joint_qd.requires_grad = True
        self.model.muscle_activation.requires_grad = True

        self.target_penalty = 1.0
        self.velocity_penalty = 0.1
        self.action_penalty = 0.0
        self.muscle_strength = 40.0

        self.discount_scale = 2.0
        self.discount_factor = 1.0


        #-----------------------
        # set up Usd renderer
        if (self.render):
            
            self.stage = Usd.Stage.CreateNew("outputs/" + self.name + ".usd")

            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0
        else:
            self.renderer = None

        self.set_target((-0.1, 0.1, 0.5), "target")

        self.integrator = df.sim.SemiImplicitIntegrator()



    def set_target(self, x, name):

        self.target = torch.tensor(x, dtype=torch.float32, device=self.adapter)

        if (self.renderer):
            self.renderer.add_sphere(self.target.tolist(), 0.05, name)

    def loss(self):

        #---------------
        # run simulation

        self.sim_time = 0.0
        
        # initial state
        self.state = self.model.state() 

        self.model.collide(self.state)

        if (self.render):
            traj = []
            for e in range(self.env_count):
                traj.append([])
 
        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        for i in range(0, self.sim_steps):

            # apply actions
            self.model.muscle_activation = (torch.tanh(4.0*self.activations[0] - 2.0)*0.5 + 0.5)*self.muscle_strength
            #self.model.muscle_activation = self.activations[0]*self.muscle_strength

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

                        for mesh, link in self.mesh_map.items():
                            
                            if link != -1:
                                X_sc = df.transform_expand(self.state.body_X_sc[link].tolist())

                                #self.renderer.add_mesh(mesh, "../assets/snu/OBJ/" + mesh + ".usd", X_sc, 1.0, self.render_time)
                                self.renderer.add_mesh(mesh, "../assets/snu/OBJ/" + mesh + ".usd", X_sc, 1.0, self.render_time)

                        for m in range(self.model.muscle_count):

                            start = self.model.muscle_start[m]
                            end = self.model.muscle_start[m+1]

                            points = []

                            for w in range(start, end):
                                
                                link = self.model.muscle_links[w]
                                point = self.model.muscle_points[w].cpu()

                                X_sc = df.transform_expand(self.state.body_X_sc[link].cpu().tolist())

                                points.append(Gf.Vec3f(df.transform_point(X_sc, point).tolist()))
                               
                            self.renderer.add_line_strip(points, name=self.muscles[m].name, radius=0.0075, color=(self.model.muscle_activation[m]/self.muscle_strength, 0.2, 0.5), time=self.render_time)

                        # render scene
                        self.render_time += self.sim_dt * self.sim_substeps
                        self.renderer.update(self.state, self.render_time)

            self.sim_time += self.sim_dt

        # loss
        if self.name == "humanoid_snu_arm":

            hand_pos = self.state.body_X_sc[self.node_map["HandR"]][0:3]

            discount_time = self.sim_time 
            discount = math.pow(self.discount_factor, discount_time*self.discount_scale)

            # loss = loss + (torch.norm(hand_pos - self.target)*self.target_penalty + 
            #                torch.norm(self.state.joint_qd)*self.velocity_penalty + 
            #                torch.norm(self.model.muscle_activation)*self.action_penalty)*discount

            #loss = loss + torch.norm(self.state.joint_qd)
            loss = loss + torch.norm(hand_pos - self.target)*self.target_penalty


        if self.name == "humanoid_snu_neck":

            # rotate a vector
            def transform_vector_torch(t, x):
                axis = t[3:6]
                w = t[6]
                return x * (2.0 *w*w - 1.0) + torch.cross(axis, x) * w * 2.0 + axis * torch.dot(axis, x) * 2.0

            forward_dir = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=self.adapter)
            up_dir = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32, device=self.adapter)
            target_dir = torch.tensor((1.0, 0.0, 0.1), dtype=torch.float32, device=self.adapter)

            head_forward = transform_vector_torch(self.state.body_X_sc[self.node_map["Head"]], forward_dir)
            head_up = transform_vector_torch(self.state.body_X_sc[self.node_map["Head"]], up_dir)

            loss = loss - torch.dot(head_forward, target_dir)*self.target_penalty - torch.dot(head_up, up_dir)*self.target_penalty

        return loss

    def run(self):

        df.config.no_grad = True

        with torch.no_grad():
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

            #print(self.activations.grad)

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
                    optimizer = torch.optim.LBFGS(params, lr=1.0, tolerance_grad=1.e-9)#, line_search_fn="strong_wolfe")

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

env = HumanoidSNU(depth=1, mode='dflex', render=True, adapter='cpu')

#df.config.no_grad = True
#df.config.check_grad = True
#df.config.verify_fp = True

#robot.load()
#env.run()

#env.load()
env.train(mode='adam')
#robot.verify(eps=1.e+1)
