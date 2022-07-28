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

import xml.etree.ElementTree as ET


class Robot:

    frame_dt = 1.0/60.0

    episode_duration = 2.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 32
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
    
    sim_time = 0.0

    train_iters = 1024
    train_rate = 0.001

    phase_count = 8
    phase_step = math.pi / phase_count * 2.0
    phase_freq = 6.0

    ground = True

    name = "humanoid"

    regularization = 1.e-3
    
    def __init__(self, depth=1, mode='numpy', render=True, adapter='cpu'):

        torch.manual_seed(42)

        builder = df.sim.ModelBuilder()

        self.adapter = adapter
        self.mode = mode
        self.render = render   

        self.parse_mjcf("./assets/" + self.name + ".xml", builder,
            stiffness=0.0,
            damping=0.0,
            contact_ke=1.e+3,
            contact_kd=1.e+3,
            contact_kf=1.e+2,
            contact_mu=0.75,
            limit_ke=1.e+2,
            limit_kd=1.e+1)

        # base transform
        
        # set joint targets to rest pose in mjcf
        if (self.name == "ant"):
            builder.joint_q[0:3] = [0.0, 0.70, 0.0]
            builder.joint_q[3:7] = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)

            builder.joint_q[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
            builder.joint_target[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]

        if (self.name == "humanoid"):
            builder.joint_q[0:3] = [0.0, 1.70, 0.0]
            builder.joint_q[3:7] = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)



        # width = 0.1
        # radius = 0.05

        # builder.add_articulation()

        # body = -1
        # for i in range(1):

        #     body = builder.add_link(
        #         parent=body, 
        #         X_pj=df.transform((2.0*width, 0.0, 0.0), df.quat_identity()), 
        #         axis=(0.0, 0.0, 1.0), 
        #         damping=0.0, 
        #         stiffness=0.0, 
        #         limit_lower=np.deg2rad(-30.0), 
        #         limit_upper=np.deg2rad(30.0), 
        #         limit_ke=100.0,
        #         limit_kd=10.0,
        #         type=df.JOINT_REVOLUTE)

        #     shape = builder.add_shape_capsule(body, pos=(width, 0.0, 0.0), half_width=width, radius=radius)

        # self.ground = False


        # finalize model
        self.model = builder.finalize(adapter)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=adapter)
        #self.model.gravity = torch.tensor((0.0, 0.0, 0.0), device=adapter)
        
        self.model.joint_q.requires_grad_()
        self.model.joint_qd.requires_grad_()

        self.network = torch.nn.Sequential(torch.nn.Linear(self.phase_count, len(self.model.joint_qd)-6, bias=False), torch.nn.Tanh()).to(adapter)

        self.action_strength = 150.0
        self.action_penalty = 0.01

        self.balance_reward = 15.0
        self.forward_reward = 1.0

        self.discount_scale = 1.0
        self.discount_factor = 0.5

        self.target = torch.tensor((0.0, 0.65, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=torch.float32, device=adapter, requires_grad=False)

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

    def parse_mjcf(
        self, 
        filename, 
        builder, 
        density=1000.0, 
        stiffness=0.0, 
        damping=0.0, 
        contact_ke=1000.0,
        contact_kd=100.0,
        contact_kf=100.0,
        contact_mu=0.5,
        limit_ke=100.0,
        limit_kd=10.0):

        file = ET.parse(filename)
        root = file.getroot()

        # map node names to link indices
        self.node_map = {}
        self.xform_map = {}
        self.mesh_map = {}
        
        type_map = { 
            "ball": df.JOINT_BALL, 
            "hinge": df.JOINT_REVOLUTE, 
            "slide": df.JOINT_PRISMATIC, 
            "free": df.JOINT_FREE, 
            "fixed": df.JOINT_FIXED
        }


        def parse_float(node, key, default):
            if key in node.attrib:
                return float(node.attrib[key])
            else:
                return default

        def parse_vec(node, key, default):
            if key in node.attrib:
                return np.fromstring(node.attrib[key], sep=" ")
            else:
                return np.array(default)

        def parse_body(body, parent):

            body_name = body.attrib["name"]
            body_pos = np.fromstring(body.attrib["pos"], sep=" ")

            #-----------------
            # add body for each joint
            
            for joint in body.findall("joint"):

                joint_name = joint.attrib["name"],
                joint_type = type_map[joint.attrib["type"]]
                joint_axis = parse_vec(joint, "axis", (0.0, 0.0, 0.0))
                joint_pos = parse_vec(joint, "pos", (0.0, 0.0, 0.0))
                joint_range = parse_vec(joint, "range", (-3.0, 3.0))
                joint_armature = parse_float(joint, "armature", 0.0)
                joint_stiffness = parse_float(joint, "stiffness", stiffness)
                joint_damping = parse_float(joint, "damping", damping)

                joint_axis = df.normalize(joint_axis)

                if (parent == -1):
                    body_pos = np.array((0.0, 0.0, 0.0))

                link = builder.add_link(
                    parent, 
                    X_pj=df.transform(body_pos, df.quat_identity()), 
                    axis=joint_axis, 
                    type=joint_type,
                    limit_lower=np.deg2rad(joint_range[0]),
                    limit_upper=np.deg2rad(joint_range[1]),
                    limit_ke=limit_ke,
                    limit_kd=limit_kd,
                    stiffness=joint_stiffness,
                    damping=joint_damping,
                    armature=joint_armature)

                parent = link
                body_pos = [0.0, 0.0, 0.0]  # todo: assumes that serial joints are all aligned at the same point

            #-----------------
            # add shapes

            for geom in body.findall("geom"):
                
                geom_name = geom.attrib["name"]
                geom_type = geom.attrib["type"]

                geom_size = parse_vec(geom, "size", [1.0])                
                geom_pos = parse_vec(geom, "pos", (0.0, 0.0, 0.0)) 
                geom_rot = parse_vec(geom, "quat", (0.0, 0.0, 0.0, 1.0))

                if (geom_type == "sphere"):

                    builder.add_shape_sphere(
                        link, 
                        pos=geom_pos, 
                        rot=geom_rot,
                        radius=geom_size[0],
                        density=density,
                        ke=contact_ke,
                        kd=contact_kd,
                        kf=contact_kf,
                        mu=contact_mu)

                elif (geom_type == "capsule"):

                    if ("fromto" in geom.attrib):
                        geom_fromto = parse_vec(geom, "fromto", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

                        start = geom_fromto[0:3]
                        end = geom_fromto[3:6]

                        # compute rotation to align dflex capsule (along x-axis), with mjcf fromto direction                        
                        axis = df.normalize(end-start)
                        angle = math.acos(np.dot(axis, (1.0, 0.0, 0.0)))
                        axis = df.normalize(np.cross(axis, (1.0, 0.0, 0.0)))

                        geom_pos = (start + end)*0.5
                        geom_rot = df.quat_from_axis_angle(axis, -angle)

                        geom_radius = geom_size[0]
                        geom_width = np.linalg.norm(end-start)*0.5

                    else:

                        geom_radius = geom_size[0]
                        geom_width = geom_size[1]
                        geom_pos = parse_vec(geom, "pos", (0.0, 0.0, 0.0))


                    builder.add_shape_capsule(
                        link,
                        pos=geom_pos,
                        rot=geom_rot,
                        radius=geom_radius,
                        half_width=geom_width,
                        density=density,
                        ke=contact_ke,
                        kd=contact_kd,
                        kf=contact_kf,
                        mu=contact_mu)
                    
                else:
                    print("Type: " + geom_type + " unsupported")
                

            #-----------------
            # recurse

            for child in body.findall("body"):
                parse_body(child, link)


        #-----------------
        # start articulation

        builder.add_articulation()

        world = root.find("worldbody")
        for body in world.findall("body"):
            parse_body(body, -1)



    def set_target(self, x, name):

        self.target = torch.tensor(x, device=self.adapter)

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
                
            # build sinusoidal input phases
            # with df.ScopedTimer("inference", False):

            #     phases = torch.zeros(self.phase_count, device=self.model.adapter)
            #     for p in range(self.phase_count):
            #         phases[p] = math.sin(self.phase_freq * self.sim_time + p * self.phase_step)

            #     # compute activations (joint torques)
            #     actions = self.network(phases) * self.action_strength


            # simulate
            with df.ScopedTimer("simulate", detailed=False, active=True):

                for i in range(0, self.sim_substeps):
                    
                    # apply actions
                    #self.state.joint_act[6:] = actions
                    self.state = self.integrator.forward(self.model, self.state, self.sim_dt, i==0)
                    self.sim_time += self.sim_dt
 
            discount_time = self.sim_time 
            discount = math.pow(self.discount_factor, discount_time*self.discount_scale)

            pos = self.state.joint_q[0:3]
            vel = df.get_body_linear_velocity(self.state.joint_qd[0:6], pos)

            loss = loss - discount*vel[0] # + torch.norm(self.state.joint_q[1]-0.5)

            # render
            with df.ScopedTimer("render", False):

                if (self.render):
                    self.render_time += self.frame_dt
                    self.renderer.update(self.state, self.render_time)

                    try:
                        self.stage.Save()
                    except:
                        print("USD save error")


        return loss

    def run(self):
        
        df.config.no_grad = True
    
        with torch.no_grad():
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

        params = self.network.parameters()

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

            #df.util.mem_report()

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
                        for p in list(params):
                            p -= self.train_rate * p.grad
                            p.grad.zero_()
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
        torch.save(self.network, "outputs/" + self.name + ".pt")

    def load(self):
        self.network = torch.load("outputs/" + self.name + ".pt")


#---------

#robot = Robot(depth=1, mode='dflex', render=True, adapter='cpu')
#robot.load()
#robot.run()

robot = Robot(depth=1, mode='dflex', render=True, adapter='cuda')
#robot.load()
#robot.train(mode='adam')
robot.run()