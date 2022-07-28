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

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# to allow tests to import the module they belong to
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

from pxr import Usd, UsdGeom, Gf

import test_util

class MultiLayerPerceptron(nn.Module):

    def __init__(self, n_in, n_out, n_hd, adapter, inference=False):

        super(MultiLayerPerceptron,self).__init__()

        self.n_in  = n_in
        self.n_out = n_out
        self.n_hd = n_hd

        #self.ll = nn.Linear(n_in, n_out)
        self.fc1 = nn.Linear(n_in, n_hd).to(adapter)
        self.fc2 = nn.Linear(n_hd, n_hd).to(adapter)
        self.fc3 = nn.Linear(n_hd, n_out).to(adapter)
        self.bn1 = nn.LayerNorm(n_in, elementwise_affine=False).to(adapter)
        self.bn2 = nn.LayerNorm(n_hd, elementwise_affine=False).to(adapter)
        self.bn3 = nn.LayerNorm(n_out, elementwise_affine=False).to(adapter)

    def forward(self, x: torch.Tensor):


        x = F.leaky_relu(self.bn2(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))

        x = torch.tanh(self.bn3(self.fc3(x))-2.0)

        return x



class HumanoidSNU:

    train_iters = 100000000
    train_rate = 0.001
    train_size = 128
    train_batch_size = 4
    train_batch_iters = 128
    train_batch_count = int(train_size/train_batch_size)
    train_data = None

    ground = True

    name = "humanoid_snu_lower" 

    regularization = 1.e-3
    
    inference = False

    initial_y = 1.0

    def __init__(self, depth=1, mode='numpy', render=True, sim_duration=1.0, adapter='cpu', inference=False):

        self.sim_duration = sim_duration      # seconds
        self.sim_substeps = 16
        self.sim_dt = (1.0 / 60.0) / self.sim_substeps
        self.sim_steps = int(self.sim_duration / self.sim_dt)
        
        self.sim_time = 0.0

        torch.manual_seed(41)
        np.random.seed(41)

        builder = df.sim.ModelBuilder()

        self.adapter = adapter
        self.mode = mode
        self.render = render

        self.filter = {}

        if self.name == "humanoid_snu_arm":
            self.filter = { "ShoulderR", "ArmR", "ForeArmR", "HandR", "Torso", "Neck" }
            self.ground = False

        if self.name == "humanoid_snu_neck":
            self.filter = { "Torso", "Neck", "Head", "ShoulderR", "ShoulderL" }
            self.ground = False

        if self.name == "humanoid_snu_lower":
            self.filter = { "Pelvis", "FemurR", "TibiaR", "TalusR", "FootThumbR", "FootPinkyR", "FemurL", "TibiaL", "TalusL", "FootThumbL", "FootPinkyL"}
            self.ground = True
            self.initial_y = 1.0

        if self.name == "humanoid_snu":
            self.filter = {}
            self.ground = True


        self.skeletons = []

        self.inference = inference
        # if (self.inference):
        #     self.train_batch_size = 1

        for i in range(self.train_batch_size):
            
            skeleton = test_util.Skeleton("assets/snu/arm.xml", "assets/snu/muscle284.xml", builder, self.filter)

            # set initial position 1m off the ground
            builder.joint_q[skeleton.coord_start + 0] = i*1.5
            builder.joint_q[skeleton.coord_start + 1] = self.initial_y
            
            # offset on z-axis
            #builder.joint_q[skeleton.coord_start + 2] = 10.0

            # initial velcoity
            #builder.joint_qd[skeleton.dof_start + 5] = 3.0

            self.skeletons.append(skeleton)

           

        # finalize model
        self.model = builder.finalize(adapter)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=adapter)
        #self.model.gravity = torch.tensor((0.0, 0.0, 0.0), device=adapter)
        
        #self.activations = torch.zeros((1, len(self.muscles)), dtype=torch.float32, device=adapter, requires_grad=True)
        #self.activations = torch.rand((1, len(self.muscles)), dtype=torch.float32, device=adapter, requires_grad=True)

        self.network = MultiLayerPerceptron(3, len(self.skeletons[0].muscles), 128, adapter)

        self.model.joint_q.requires_grad = True
        self.model.joint_qd.requires_grad = True
        self.model.muscle_activation.requires_grad = True

        self.target_penalty = 1.0
        self.velocity_penalty = 0.1
        self.action_penalty = 0.0
        self.muscle_strength = 40.0

        self.discount_scale = 2.0
        self.discount_factor = 1.0

        # generate training data
        targets = []
        for i in range(self.train_size):
            
            # generate a random point in -1, 1 away from the head
            t = np.random.rand(2)*2.0 - 1.0
            t[1] += 0.5

            targets.append((t[0], t[1] + 0.5, 1.0))

        self.train_data = torch.tensor(targets, dtype=torch.float32, device=self.adapter)

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

        self.set_target(torch.tensor((0.75, 0.4, 0.5), dtype=torch.float32, device=self.adapter), "target")

        self.integrator = df.sim.SemiImplicitIntegrator()



    def set_target(self, x, name):

        self.target = x

        if (self.renderer):
            self.renderer.add_sphere(self.target.tolist(), 0.05, name, self.render_time)

    def loss(self):

        #---------------
        # run simulation

        self.sim_time = 0.0

        # initial state
        self.state = self.model.state() 
 
        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        # apply actions
        
        #self.model.muscle_activation = self.activations[0]*self.muscle_strength
        
        # compute activations for each target in the batch
        targets = self.train_data[0:self.train_batch_size]
        activations = torch.flatten(self.network(targets))

        self.model.muscle_activation = (activations*0.5 + 0.5)*self.muscle_strength

        # one time collision 
        self.model.collide(self.state)

        for i in range(self.sim_steps):

            # apply random actions per-frame
            #self.model.muscle_activation = (activations*0.5 + 0.5 + torch.rand_like(activations,dtype=torch.float32, device=self.model.adapter))*self.muscle_strength

            # simulate
            with df.ScopedTimer("fd", detailed=False, active=False):
                self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

            #if self.inference:
                
                #x = math.cos(self.sim_time*0.5)*0.5
                #y = math.sin(self.sim_time*0.5)*0.5

                # t = self.sim_time*0.5
                # x = math.sin(t)*0.5
                # y = math.sin(t)*math.cos(t)*0.5

                # self.set_target(torch.tensor((x, y + 0.5, 1.0), dtype=torch.float32, device=self.adapter), "target")

                # activations = self.network(self.target)
                # self.model.muscle_activation = (activations*0.5 + 0.5)*self.muscle_strength

            # render
            with df.ScopedTimer("render", False):
                if (self.render and (i % self.sim_substeps == 0)):

                    with torch.no_grad():

                        muscle_start = 0
                        skel_index = 0

                        for s in self.skeletons:
                            for mesh, link in s.mesh_map.items():
                                
                                if link != -1:
                                    X_sc = df.transform_expand(self.state.body_X_sc[link].tolist())

                                    #self.renderer.add_mesh(mesh, "../assets/snu/OBJ/" + mesh + ".usd", X_sc, 1.0, self.render_time)
                                    self.renderer.add_mesh(mesh, "../assets/snu/OBJ/" + mesh + ".usd", X_sc, 1.0, self.render_time)

                            for m in range(len(s.muscles)):#.self.model.muscle_count):

                                start = self.model.muscle_start[muscle_start + m].item()
                                end = self.model.muscle_start[muscle_start + m + 1].item()

                                points = []

                                for w in range(start, end):
                                    
                                    link = self.model.muscle_links[w].item()
                                    point = self.model.muscle_points[w].cpu().numpy()

                                    X_sc = df.transform_expand(self.state.body_X_sc[link].cpu().tolist())

                                    points.append(Gf.Vec3f(df.transform_point(X_sc, point).tolist()))
                                
                                self.renderer.add_line_strip(points, name=s.muscles[m].name + str(skel_index), radius=0.0075, color=(self.model.muscle_activation[muscle_start + m]/self.muscle_strength, 0.2, 0.5), time=self.render_time)
                            
                            muscle_start += len(s.muscles)
                            skel_index += 1

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

            for i in range(self.train_batch_size):
                       
                skel = self.skeletons[i]
            
                head_pos = self.state.body_X_sc[skel.node_map["Head"]][0:3]
                head_forward = transform_vector_torch(self.state.body_X_sc[skel.node_map["Head"]], forward_dir)
                head_up = transform_vector_torch(self.state.body_X_sc[skel.node_map["Head"]], up_dir)

                target_dir = self.train_data[i] - head_pos

                loss_forward = torch.dot(head_forward, target_dir)*self.target_penalty
                loss_up = torch.dot(head_up, up_dir)*self.target_penalty*0.5
                loss_penalty = torch.dot(activations, activations)*self.action_penalty

                loss = loss - loss_forward - loss_up + loss_penalty
            
            #self.writer.add_scalar("loss_forward", loss_forward.item(), self.step_count)
            #self.writer.add_scalar("loss_up", loss_up.item(), self.step_count)
            #self.writer.add_scalar("loss_penalty", loss_penalty.item(), self.step_count)


        return loss

    def run(self):

        df.config.no_grad = True

        self.inference = True

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

        self.writer = SummaryWriter()
        self.writer.add_hparams({"lr": self.train_rate, "mode": mode}, {})

        # param to train
        self.step_count = 0
        self.best_loss = math.inf

        optimizer = None
        scheduler = None

        params = self.network.parameters()#[self.activations]

        def closure():

            batch = int(self.step_count/self.train_batch_iters)%self.train_batch_count

            print("Batch: " + str(batch) + " Iter: " + str(self.step_count%self.train_batch_iters))

            if (optimizer):
                optimizer.zero_grad()

            # compute loss on all examples
            with df.ScopedTimer("forward"):#, detailed=True):
                l = self.loss()

            # compute gradient
            with df.ScopedTimer("backward"):#, detailed=True):
                l.backward()

            # batch stats
            self.writer.add_scalar("loss_batch", l.item(), self.step_count)
            self.writer.flush()

            print(str(self.step_count) + ": " + str(l))
            self.step_count += 1

            with df.ScopedTimer("save"):
                try:
                    self.stage.Save()
                except:
                    print("USD save error")

                # save network
                if (l < self.best_loss):
                    self.save()
                    self.best_loss = l

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
                    last_LR = 1e-5
                    init_LR = 1e-3
                    decay_LR_steps = 2000
                    gamma = math.exp(math.log(last_LR/init_LR)/decay_LR_steps)
                    
                    optimizer = torch.optim.Adam(params, lr=self.train_rate, weight_decay=1e-5)
                    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)

                # SGD
                if (mode == 'sgd'):
                    optimizer = torch.optim.SGD(params, lr=self.train_rate, momentum=0.8, nesterov=True)

                # train
                for i in range(self.train_iters):
                    
                    print("Step: " + str(i))
                    
                    if optimizer:
                        optimizer.step(closure)
                    
                    if scheduler:
                        scheduler.step() 

                # final save
                try:
                    self.stage.Save()
                except:
                    print("USD save error")

    def save(self):
        torch.save(self.network, "outputs/" + self.name + ".pt")

    def load(self, suffix=""):
        self.network = torch.load("outputs/" + self.name + suffix + ".pt")
        
        if self.inference:
            self.network.eval()
        else:
            self.network.train()


#---------

#env = HumanoidSNU(depth=1, mode='dflex', render=True, sim_duration=2.0, adapter='cuda')
#env.train(mode='adam')

env = HumanoidSNU(depth=1, mode='dflex', render=True, sim_duration=2.0, adapter='cuda', inference=True)
#env.load()
env.run()

