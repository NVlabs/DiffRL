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

from pxr import Usd, UsdGeom, Gf


use_omni = False
if use_omni:
    import omni.usd


class Experiment:

    name = "kit_walker"

    network_file = None
    record = True

    render_time = 0.0
    render_enabled = True
    
    def __init__(self):
        pass 

    def reset(self, adapter='cuda'):

        self.episode_duration = 5.0       # seconds

        self.frame_dt = 1.0/60.0
        self.frame_count = int(self.episode_duration/self.frame_dt)

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.train_max_iters = 10000
        self.train_iter = 0
        self.train_rate = 0.025
        self.train_loss = []
        self.train_loss_best = math.inf

        self.phase_count = 8
        self.phase_step = math.pi / self.phase_count * 2.0
        self.phase_freq = 5.0

        self.render_time = 0.0

        torch.manual_seed(42)

        builder = df.sim.ModelBuilder()

        self.train_loss = []
        self.optimizer = None


        #mesh = Usd.Stage.Open("assets/prop.usda")
        if use_omni == False:
            stage = Usd.Stage.Open("kit_walker.usda")            
        else:
            stage = omni.usd.get_context().get_stage()
            
        # ostrich
        # geom = UsdGeom.Mesh(stage.GetPrimAtPath("/ostrich"))
        # points = geom.GetPointsAttr().Get()

        # builder.add_soft_mesh(pos=(0.0, 0.0, 0.0),
        #                       rot=df.quat_identity(),
        #                       scale=2.0,
        #                       vel=(0.0, 0.0, 0.0),
        #                       vertices=points,
        #                       indices=tet_indices,
        #                       density=1.0,
        #                       k_mu=2000.0,
        #                       k_lambda=2000.0,
        #                       k_damp=1.0)

        # bear
        geom = UsdGeom.Mesh(stage.GetPrimAtPath("/bear"))
        points = geom.GetPointsAttr().Get()

        xform = geom.ComputeLocalToWorldTransform(0.0)
        for i in range(len(points)):
            points[i] = xform.Transform(points[i])

        tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()
        tri_indices = geom.GetFaceVertexIndicesAttr().Get()
        tri_counts = geom.GetFaceVertexCountsAttr().Get()

        builder.add_soft_mesh(pos=(0.0, 0.0, 0.0),
                              rot=df.quat_identity(),
                              scale=2.0,
                              vel=(0.0, 0.0, 0.0),
                              vertices=points,
                              indices=tet_indices,
                              density=1.0,
                              k_mu=2000.0,
                              k_lambda=2000.0,
                              k_damp=2.0)

        # # table       
        # geom = UsdGeom.Mesh(stage.GetPrimAtPath("/table"))
        # points = geom.GetPointsAttr().Get()

        # builder.add_soft_mesh(pos=(0.0, 0.0, 0.0),
        #                       rot=df.quat_identity(),
        #                       scale=1.0,
        #                       vel=(0.0, 0.0, 0.0),
        #                       vertices=points,
        #                       indices=tet_indices,
        #                       density=1.0,
        #                       k_mu=1000.0,
        #                       k_lambda=1000.0,
        #                       k_damp=1.0)                              



        #builder.add_soft_grid(pos=(0.0, 0.5, 0.0), rot=(0.0, 0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), dim_x=1, dim_y=2, dim_z=1, cell_x=0.5, cell_y=0.5, cell_z=0.5, density=1.0)

        # s = 2.0
        # builder.add_particle((0.0, 0.5, 0.0), (0.0, 0.0, 0.0), 1.0)
        # builder.add_particle((s,  0.5, 0.0), (0.0, 0.0, 0.0), 1.0)
        # builder.add_particle((0.0, 0.5, s), (0.0, 0.0, 0.0), 1.0)
        # builder.add_particle((0.0, s + 0.5, 0.0), (0.0, 0.0, 0.0), 1.0)

        # builder.add_tetrahedron(1, 3, 0, 2)

        self.model = builder.finalize(adapter)

        #self.model.tet_kl = 1000.0
        #self.model.tet_km = 1000.0
        #self.model.tet_kd = 1.0

        # disable triangle dynamics (just used for rendering)
        self.model.tri_ke = 0.0
        self.model.tri_ka = 0.0
        self.model.tri_kd = 0.0
        self.model.tri_kb = 0.0

        self.model.contact_ke = 1.e+3*2.0
        self.model.contact_kd = 0.1
        self.model.contact_kf = 10.0
        self.model.contact_mu = 0.7

        self.model.particle_radius = 0.05
        self.model.ground = True

        #self.model.gravity = torch.tensor((0.0, -1.0, 0.0), device=adapter)

        # one fully connected layer + tanh activation
        self.network = torch.nn.Sequential(torch.nn.Linear(self.phase_count, self.model.tet_count, bias=False), torch.nn.Tanh()).to(adapter)

        self.activation_strength = 0.3
        self.activation_penalty = 0.0

        #-----------------------
        # set up Usd renderer

        self.stage = stage#Usd.Stage.CreateNew("outputs/fem.usd")

        if (self.stage):
            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

        self.integrator = df.sim.SemiImplicitIntegrator()
        self.state = self.model.state()

        if self.network_file:
            self.load(self.network_file)

    def inference(self):
        # build sinusoidal input phases
        with df.ScopedTimer("inference", False):
            phases = torch.zeros(self.phase_count, device=self.model.adapter)
            for p in range(self.phase_count):
                phases[p] = math.sin(self.phase_freq*self.sim_time + p * self.phase_step)

            # compute activations 
            self.model.tet_activations = self.network(phases) * self.activation_strength


    def simulate(self, no_grad=False):

        # set grad mode
        df.config.no_grad = no_grad

        for i in range(self.sim_substeps):
            self.state = self.integrator.forward(self.model, self.state, self.sim_dt)
            self.sim_time += self.sim_dt
            

    def render(self):

        with df.ScopedTimer("render", False):

            if (self.record):
                self.render_time += self.frame_dt

            if (self.stage):
                self.renderer.update(self.state, self.render_time)



    def loss(self, render=True):

        #-----------------------
        # run simulation
        self.sim_time = 0.0

        self.state = self.model.state()

        loss = torch.zeros(1, requires_grad=True, device=self.model.adapter)

        for f in range(self.frame_count):

            self.inference()
            self.simulate()
            
            if (self.render_enabled):
                self.render()
            
            # loss
            with df.ScopedTimer("loss", False):
                com_loss = torch.mean(self.state.particle_qd, 0)
                #act_loss = torch.norm(self.model.tet_activations)*self.activation_penalty

                loss = loss - com_loss[0] + torch.norm(com_loss[1])  + torch.norm(com_loss[2])# + act_loss

        return loss

    def run(self, profile=False):

        self.inference()
        self.simulate(no_grad=True)

        if (self.render_enabled):
            self.render()


    def train(self, mode='gd'):

        # create optimizer if requested
        if (self.optimizer == None):
        
            # L-BFGS
            if (mode == 'lbfgs'):
                self.optimizer = torch.optim.LBFGS(self.network.parameters(), lr=1.0, tolerance_grad=1.e-5, tolerance_change=0.01, line_search_fn="strong_wolfe")

            # Adam
            if (mode == 'adam'):
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.train_rate)

            # SGD
            if (mode == 'sgd'):
                self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.train_rate, momentum=0.5, nesterov=True)

        # closure for evaluating loss (called by optimizers)
        def closure():

            if (self.optimizer):
                self.optimizer.zero_grad()

            # render every N steps
            render = True

            with df.ScopedTimer("forward"):
                l = self.loss(render)

            # save best network so far
            if (l < self.train_loss_best):
                self.train_loss_best = float(l)
                self.save()

            self.train_loss.append(float(l))
            
            df.log("Iteration: {} Loss: {}".format(len(self.train_loss), l.item()))

            # save USD file
            if use_omni == False:
                try:
                    self.stage.Save()
                except:
                    print("Usd save error")

            # calculate gradient
            with df.ScopedTimer("backward"):
                l.backward()

            return l

        # perform optimization step
        with df.ScopedTimer("step"):

            if (mode == 'gd'):

                # simple Gradient Descent
                closure()

                with torch.no_grad():
                    params = self.network.parameters()

                    for p in params:
                        if p.grad is None:
                            continue

                        p -= self.train_rate * p.grad
                        p.grad.zero_()
            else:
 
                self.optimizer.step(closure)

        self.train_iter += 1


    def save(self):
        torch.save(self.network, self.name + str(self.train_iter) + ".pt")

    def load(self, file):
        self.network = torch.load(file)
        self.network.eval()

        df.log("Loaded pretrained network: " + file)


#---------

experiment = Experiment()

if use_omni == False:
    experiment.reset(adapter='cuda')
    #experiment.load("kit_walker19.pt")
    #experiment.train_iter = 19

    # with df.ScopedTimer("update", detailed=False):
    #     for i in range(experiment.frame_count):
    #         experiment.run()

    # experiment.stage.Save()

    experiment.render_enabled = False
    
    #with torch.autograd.profiler.profile() as prof:
    with df.ScopedTimer("train", detailed=True):
        #for i in range(experiment.train_max_iters):
        experiment.train('adam')

    #print(prof.key_averages().table())