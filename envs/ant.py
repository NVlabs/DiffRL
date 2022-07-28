# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from envs.dflex_env import DFlexEnv
import math
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

from utils import load_utils as lu
from utils import torch_utils as tu


class AntEnv(DFlexEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=4096, seed=0, episode_length=1000, no_grad=True, stochastic_init=False, MM_caching_frequency = 1, early_termination = True):
        num_obs = 37
        num_act = 8
    
        super(AntEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)

        self.stochastic_init = stochastic_init
        self.early_termination = early_termination

        self.init_sim()

        # other parameters
        self.termination_height = 0.27
        self.action_strength = 200.0
        self.action_penalty = 0.0
        self.joint_vel_obs_scaling = 0.1

        #-----------------------
        # set up Usd renderer
        if (self.visualize):
            self.stage = Usd.Stage.CreateNew("outputs/" + "Ant_" + str(self.num_envs) + ".usd")

            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0/60.0
        self.sim_substeps = 16
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 15
        self.num_joint_qd = 14

        self.x_unit_tensor = tu.to_torch([1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch([0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch([0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        self.start_rot = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)
        self.start_rotation = tu.to_torch(self.start_rot, device=self.device, requires_grad=False)

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = tu.to_torch([10000.0, 0.0, 0.0], device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        self.start_pos = []
        self.start_joint_q = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
        self.start_joint_target = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]

        if self.visualize:
            self.env_dist = 2.5
        else:
            self.env_dist = 0. # set to zero for training for numerical consistency

        start_height = 0.75

        asset_folder = os.path.join(os.path.dirname(__file__), 'assets')
        for i in range(self.num_environments):
            lu.parse_mjcf(os.path.join(asset_folder, "ant.xml"), self.builder,
                density=1000.0,
                stiffness=0.0,
                damping=1.0,
                contact_ke=4.e+4,
                contact_kd=1.e+4,
                contact_kf=3.e+3,
                contact_mu=0.75,
                limit_ke=1.e+3,
                limit_kd=1.e+1,
                armature=0.05)

            # base transform
            start_pos_z = i*self.env_dist
            self.start_pos.append([0.0, start_height, start_pos_z])

            self.builder.joint_q[i*self.num_joint_q:i*self.num_joint_q + 3] = self.start_pos[-1]
            self.builder.joint_q[i*self.num_joint_q + 3:i*self.num_joint_q + 7] = self.start_rot

            # set joint targets to rest pose in mjcf
            self.builder.joint_q[i*self.num_joint_q + 7:i*self.num_joint_q + 15] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
            self.builder.joint_target[i*self.num_joint_q + 7:i*self.num_joint_q + 15] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]

        self.start_pos = tu.to_torch(self.start_pos, device=self.device)
        self.start_joint_q = tu.to_torch(self.start_joint_q, device=self.device)
        self.start_joint_target = tu.to_torch(self.start_joint_target, device=self.device)

        # finalize model
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=self.device)

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()

        if (self.model.ground):
            self.model.collide(self.state)

    def render(self, mode = 'human'):
        if self.visualize:
            self.render_time += self.dt
            self.renderer.update(self.state, self.render_time)

            render_interval = 1
            if (self.num_frames == render_interval):
                try:
                    self.stage.Save()
                except:
                    print("USD save error")

                self.num_frames -= render_interval

    def step(self, actions):
        actions = actions.view((self.num_envs, self.num_actions))

        actions = torch.clip(actions, -1., 1.)

        self.actions = actions.clone()

        self.state.joint_act.view(self.num_envs, -1)[:, 6:] = actions * self.action_strength
        
        self.state = self.integrator.forward(self.model, self.state, self.sim_dt, self.sim_substeps, self.MM_caching_frequency)
        self.sim_time += self.sim_dt

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'episode_end': self.termination_buf
                }

        if len(env_ids) > 0:
           self.reset(env_ids)

        self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def reset(self, env_ids = None, force_reset = True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # clone the state to avoid gradient error
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()

            # fixed start state
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] = self.start_pos[env_ids, :].clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7] = self.start_rotation.clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] = self.start_joint_q.clone()
            self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.

            # randomization
            if self.stochastic_init:
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] = self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] + 0.1 * (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.
                angle = (torch.rand(len(env_ids), device = self.device) - 0.5) * np.pi / 12.
                axis = torch.nn.functional.normalize(torch.rand((len(env_ids), 3), device = self.device) - 0.5)
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7] = tu.quat_mul(self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7], tu.quat_from_angle_axis(angle, axis))
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] = self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] + 0.2 * (torch.rand(size=(len(env_ids), self.num_joint_q - 7), device = self.device) - 0.5) * 2.
                self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.5 * (torch.rand(size=(len(env_ids), 14), device=self.device) - 0.5)

            # clear action
            self.actions = self.actions.clone()
            self.actions[env_ids, :] = torch.zeros((len(env_ids), self.num_actions), device = self.device, dtype = torch.float)

            self.progress_buf[env_ids] = 0

            self.calculateObservations()

        return self.obs_buf
    
    '''
    cut off the gradient from the current state to previous states
    '''
    def clear_grad(self, checkpoint = None):
        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}
                checkpoint['joint_q'] = self.state.joint_q.clone()
                checkpoint['joint_qd'] = self.state.joint_qd.clone()
                checkpoint['actions'] = self.actions.clone()
                checkpoint['progress_buf'] = self.progress_buf.clone()

            current_joint_q = checkpoint['joint_q'].clone()
            current_joint_qd = checkpoint['joint_qd'].clone()
            self.state = self.model.state()
            self.state.joint_q = current_joint_q
            self.state.joint_qd = current_joint_qd
            self.actions = checkpoint['actions'].clone()
            self.progress_buf = checkpoint['progress_buf'].clone()

    '''
    This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and it returns the observation vectors
    '''
    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()

        return self.obs_buf

    def get_checkpoint(self):
        checkpoint = {}
        checkpoint['joint_q'] = self.state.joint_q.clone()
        checkpoint['joint_qd'] = self.state.joint_qd.clone()
        checkpoint['actions'] = self.actions.clone()
        checkpoint['progress_buf'] = self.progress_buf.clone()

        return checkpoint

    def calculateObservations(self):
        torso_pos = self.state.joint_q.view(self.num_envs, -1)[:, 0:3]
        torso_rot = self.state.joint_q.view(self.num_envs, -1)[:, 3:7]
        lin_vel = self.state.joint_qd.view(self.num_envs, -1)[:, 3:6]
        ang_vel = self.state.joint_qd.view(self.num_envs, -1)[:, 0:3]

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(torso_pos, ang_vel, dim = -1)

        to_target = self.targets + self.start_pos - torso_pos
        to_target[:, 1] = 0.0
        
        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot)

        up_vec = tu.quat_rotate(torso_quat, self.basis_vec1)
        heading_vec = tu.quat_rotate(torso_quat, self.basis_vec0)

        self.obs_buf = torch.cat([torso_pos[:, 1:2], # 0
                                torso_rot, # 1:5
                                lin_vel, # 5:8
                                ang_vel, # 8:11
                                self.state.joint_q.view(self.num_envs, -1)[:, 7:], # 11:19
                                self.joint_vel_obs_scaling * self.state.joint_qd.view(self.num_envs, -1)[:, 6:], # 19:27
                                up_vec[:, 1:2], # 27
                                (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1), # 28
                                self.actions.clone()], # 29:37
                                dim = -1)

    def calculateReward(self):
        up_reward = 0.1 * self.obs_buf[:, 27]
        heading_reward = self.obs_buf[:, 28]
        height_reward = self.obs_buf[:, 0] - self.termination_height

        progress_reward = self.obs_buf[:, 5]

        self.rew_buf = progress_reward + up_reward + heading_reward + height_reward + torch.sum(self.actions ** 2, dim = -1) * self.action_penalty

        # reset agents
        if self.early_termination:
            self.reset_buf = torch.where(self.obs_buf[:, 0] < self.termination_height, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)