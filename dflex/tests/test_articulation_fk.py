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
np.set_printoptions(precision=5, linewidth=256, suppress=True)

from pxr import Usd, UsdGeom, Gf

import test_util


class Robot:

    sim_duration = 10.0      # seconds
    sim_substeps = 4
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 64
    train_rate = 0.05        #1.0/(sim_dt*sim_dt)

    def __init__(self, adapter='cpu'):

        torch.manual_seed(42)

        builder = df.sim.ModelBuilder()

        x = 0.0
        w = 0.5

        max_depth = 3

        # create a branched tree
        builder.add_articulation()
        test_util.build_tree(builder, angle=0.0, width=w, max_depth=max_depth)
  
        # add weight
        if (True):
            
            radius = 0.1

            X_pj = df.transform((w * 2.0, 0.0, 0.0), df.quat_from_axis_angle( (0.0, 0.0, 1.0), 0.0))
            X_cm = df.transform((radius, 0.0, 0.0), df.quat_identity())

            parent = len(builder.body_mass)-1
            link = builder.add_link(parent, X_pj, (0.0, 0.0, 1.0), df.JOINT_REVOLUTE)
            shape = builder.add_shape_sphere(link, pos=(0.0, 0.0, 0.0), radius=radius)

        self.model = builder.finalize(adapter)
        self.model.contact_ke = 1.e+4
        self.model.contact_kd = 1000.0
        self.model.contact_kf = 100.0
        self.model.contact_mu = 0.75
        self.model.ground = False
        self.model.gravity = torch.tensor((0.0, 0.0, 0.0), device=adapter)

        # base state
        self.state = self.model.state()
        self.state.joint_q.requires_grad_()

        # ik target        
        self.target = torch.tensor((1.0, 2.0, 0.0), device=adapter)

        #-----------------------
        # set up Usd renderer

        self.stage = Usd.Stage.CreateNew("outputs/articulation_fk.usda")

        if (self.stage):
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

        if (True):

            self.state.body_X_sc, self.state.body_X_sm = df.adjoint.launch(df.eval_rigid_fk,
                    1,
                    [   # inputs
                        self.model.articulation_start,
                        self.model.joint_type,
                        self.model.joint_parent,
                        self.model.joint_q_start,
                        self.model.joint_qd_start,
                        self.state.joint_q,
                        self.model.joint_X_pj,
                        self.model.joint_X_cm,
                        self.model.joint_axis
                    ], 
                    [   # outputs
                        self.state.body_X_sc, 
                        self.state.body_X_sm
                    ],
                    adapter='cpu',
                    preserve_output=True)


            p = self.state.body_X_sm[3][0:3]

            err = torch.norm(p - self.target)

            # try:
            #     art_start = self.art.articulation_start.clone()
            #     art_end = self.art.articulation_end.clone()
            #     joint_type = self.art.joint_type.clone()
            #     joint_parent = self.art.joint_parent.clone()
            #     joint_q_start = self.art.joint_q_start.clone()
            #     joint_qd_start = self.art.joint_qd_start.clone()
            #     joint_q = self.art.joint_q.clone()
            #     joint_X_pj = self.art.joint_X_pj.clone()
            #     joint_X_cm = self.art.joint_X_cm.clone()
            #     joint_axis = self.art.joint_axis.clone()

            #     torch.autograd.gradcheck(df.EvalRigidFowardKinematicsFunc.apply, (
            #         art_start,
            #         art_end,
            #         joint_type,
            #         joint_parent,
            #         joint_q_start,
            #         joint_qd_start,
            #         joint_q,
            #         joint_X_pj,
            #         joint_X_cm,
            #         joint_axis,
            #         'cpu'), eps=1e-3, atol=1e-3, raise_exception=True)

            # except Exception as e:
            #     print("failed: " + str(e))

            # render
            with df.ScopedTimer("render", False):
                if (self.stage):
                    self.render_time += self.sim_dt * self.sim_substeps
                    self.renderer.update(self.state, self.render_time)
                    #self.stage.Save()

            self.sim_time += self.sim_dt

            return err


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

        params = [self.state.joint_q]

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
                    optimizer = torch.optim.LBFGS(params, lr=0.2, tolerance_grad=1.e-5, tolerance_change=0.01, line_search_fn="strong_wolfe")

                # Adam
                if (mode == 'adam'):
                    optimizer = torch.optim.Adam(params, lr=self.train_rate)

                # SGD
                if (mode == 'sgd'):
                    optimizer = torch.optim.SGD(params, lr=self.train_rate, momentum=0.8)

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

robot = Robot(adapter='cpu')
#robot.run()
mode = 'lbfgs'

robot.set_target((1.0, 2.0, 0.0), "target_1")
robot.train(mode)

robot.set_target((1.0, -2.0, 0.0), "target_2")
robot.train(mode)

robot.set_target((-1.0, -2.0, 0.0), "target_3")
robot.train(mode)

robot.set_target((-2.0, 2.0, 0.0), "target_4")
robot.train(mode)

#rigid.stage.Save()
