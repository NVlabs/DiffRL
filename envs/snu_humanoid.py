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
    from pxr import Usd, UsdGeom, Gf
except ModuleNotFoundError:
    print("No pxr package")

from utils import load_utils as lu
from utils import torch_utils as tu


class SNUHumanoidEnv(DFlexEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=4096, seed=0, episode_length=1000, no_grad=True, stochastic_init=False, MM_caching_frequency = 1):

        self.filter = { "Pelvis", "FemurR", "TibiaR", "TalusR", "FootThumbR", "FootPinkyR", "FemurL", "TibiaL", "TalusL", "FootThumbL", "FootPinkyL"}

        self.skeletons = []
        self.muscle_strengths = []

        self.mtu_actuations = True 

        self.inv_control_freq = 1

        # "humanoid_snu_lower"
        self.num_joint_q = 29
        self.num_joint_qd = 24

        self.num_dof = self.num_joint_q - 7 # 22
        self.num_muscles = 152

        self.str_scale = 0.6

        num_act = self.num_joint_qd - 6 # 18
        num_obs = 71 # 13 + 22 + 18 + 18

        if self.mtu_actuations:
            num_obs = 53 # 71 - 18

        if self.mtu_actuations:
            num_act = self.num_muscles
        
        super(SNUHumanoidEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)

        self.stochastic_init = stochastic_init

        self.init_sim()

        # other parameters
        self.termination_height = 0.46
        self.termination_tolerance = 0.05
        self.height_rew_scale = 4.0
        self.action_strength = 100.0
        self.action_penalty = -0.001
        self.joint_vel_obs_scaling = 0.1

        #-----------------------
        # set up Usd renderer
        if (self.visualize):
            self.stage = Usd.Stage.CreateNew("outputs/" + self.name + "HumanoidSNU_Low_" + str(self.num_envs) + ".usd")

            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0/60.0
        self.sim_substeps = 48

        self.sim_dt = self.dt

        self.ground = True

        self.x_unit_tensor = tu.to_torch([1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch([0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch([0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        self.start_rot = df.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.5)
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

        if self.visualize:
            self.env_dist = 2.0
        else:
            self.env_dist = 0. # set to zero for training for numerical consistency

        start_height = 1.0

        self.asset_folder = os.path.join(os.path.dirname(__file__), 'assets/snu')
        asset_path = os.path.join(self.asset_folder, "human.xml")
        muscle_path = os.path.join(self.asset_folder, "muscle284.xml")

        for i in range(self.num_environments):
            
            if self.mtu_actuations:
                skeleton = lu.Skeleton(asset_path, muscle_path, self.builder, self.filter, 
                                        stiffness=5.0, 
                                        damping=2.0, 
                                        contact_ke=5e3,
                                        contact_kd=2e3,
                                        contact_kf=1e3,
                                        contact_mu=0.5,
                                        limit_ke=1e3,
                                        limit_kd=1e1,
                                        armature=0.05)
            else:
                skeleton = lu.Skeleton(asset_path, None, self.builder, self.filter,
                                        stiffness=5.0, 
                                        damping=2.0, 
                                        contact_ke=5e3,
                                        contact_kd=2e3,
                                        contact_kf=1e3,
                                        contact_mu=0.5,
                                        limit_ke=1e3,
                                        limit_kd=1e1,
                                        armature=0.05)

            # set initial position 1m off the ground
            self.builder.joint_q[skeleton.coord_start + 2] = i * self.env_dist
            self.builder.joint_q[skeleton.coord_start + 1] = start_height

            self.builder.joint_q[skeleton.coord_start + 3:skeleton.coord_start + 7] = self.start_rot

            self.start_pos.append([self.builder.joint_q[skeleton.coord_start], start_height, self.builder.joint_q[skeleton.coord_start + 2]])

            self.skeletons.append(skeleton)

        num_muscles = len(self.skeletons[0].muscles)
        num_q = int(len(self.builder.joint_q)/self.num_environments)
        num_qd = int(len(self.builder.joint_qd)/self.num_environments)
        print(num_q, num_qd)

        print("Start joint_q: ", self.builder.joint_q[0:num_q])
        print("Num muscles: ", num_muscles)

        self.start_joint_q = self.builder.joint_q[7:num_q].copy()
        self.start_joint_target = self.start_joint_q.copy()
        
        for m in self.skeletons[0].muscles:
            self.muscle_strengths.append(self.str_scale * m.muscle_strength)

        for mi in range(len(self.muscle_strengths)):
            self.muscle_strengths[mi] = self.str_scale * self.muscle_strengths[mi]

        self.muscle_strengths = tu.to_torch(self.muscle_strengths, device=self.device).repeat(self.num_envs)
    
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
            with torch.no_grad():

                muscle_start = 0
                skel_index = 0

                for s in self.skeletons:
                    for mesh, link in s.mesh_map.items():
                        
                        if link != -1:
                            X_sc = df.transform_expand(self.state.body_X_sc[link].tolist())

                            mesh_path = os.path.join(self.asset_folder, "OBJ/" + mesh + ".usd")

                            self.renderer.add_mesh(mesh, mesh_path, X_sc, 1.0, self.render_time)

                    for m in range(len(s.muscles)):

                        start = self.model.muscle_start[muscle_start + m].item()
                        end = self.model.muscle_start[muscle_start + m + 1].item()

                        points = []

                        for w in range(start, end):
                            
                            link = self.model.muscle_links[w].item()
                            point = self.model.muscle_points[w].cpu().numpy()

                            X_sc = df.transform_expand(self.state.body_X_sc[link].cpu().tolist())

                            points.append(Gf.Vec3f(df.transform_point(X_sc, point).tolist()))
                        
                        self.renderer.add_line_strip(points, name=s.muscles[m].name + str(skel_index), radius=0.0075, color=(self.model.muscle_activation[muscle_start + m]/self.muscle_strengths[m], 0.2, 0.5), time=self.render_time)
                    
                    muscle_start += len(s.muscles)
                    skel_index += 1

            self.render_time += self.dt * self.inv_control_freq
            self.renderer.update(self.state, self.render_time)

            if (self.num_frames == 1):
                try:
                    self.stage.Save()
                except:
                    print("USD save error")

                self.num_frames -= 1

    def step(self, actions):
        actions = actions.view((self.num_envs, self.num_actions))

        actions = torch.clip(actions, -1., 1.)
        actions = actions * 0.5 + 0.5
    
        ##### an ugly fix for simulation nan values #### # reference: https://github.com/pytorch/pytorch/issues/15131
        def create_hook():
            def hook(grad):
                torch.nan_to_num(grad, 0.0, 0.0, 0.0, out = grad)
            return hook
        
        if self.state.joint_q.requires_grad:
            self.state.joint_q.register_hook(create_hook())
        if self.state.joint_qd.requires_grad:
            self.state.joint_qd.register_hook(create_hook())
        if actions.requires_grad:
            actions.register_hook(create_hook())
        #################################################

        self.actions = actions.clone()

        for ci in range(self.inv_control_freq):
            if self.mtu_actuations:
                self.model.muscle_activation = actions.view(-1) * self.muscle_strengths
            else:
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

        with df.ScopedTimer("render", False):
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
                self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.5 * (torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5)

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
                checkpoint = {} # NOTE: any other things to restore?
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
                                self.state.joint_q.view(self.num_envs, -1)[:, 7:], # 11:33
                                self.joint_vel_obs_scaling * self.state.joint_qd.view(self.num_envs, -1)[:, 6:], # 33:51
                                up_vec[:, 1:2], # 51
                                (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1)], # 52
                                dim = -1)

    def calculateReward(self):
        up_reward = 0.1 * self.obs_buf[:, 51]
        heading_reward = self.obs_buf[:, 52]

        height_diff = self.obs_buf[:, 0] - (self.termination_height + self.termination_tolerance)
        height_reward = torch.clip(height_diff, -1.0, self.termination_tolerance)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward) # JIE: not smooth
        height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)
        
        act_penalty = torch.sum(torch.abs(self.actions), dim = -1) * self.action_penalty #torch.sum(self.actions ** 2, dim = -1) * self.action_penalty

        progress_reward = self.obs_buf[:, 5]

        self.rew_buf = progress_reward + up_reward + heading_reward + act_penalty
        
        # reset agents
        self.reset_buf = torch.where(self.obs_buf[:, 0] < self.termination_height, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # an ugly fix for simulation nan values
        nan_masks = torch.logical_or(torch.isnan(self.obs_buf).sum(-1) > 0, torch.logical_or(torch.isnan(self.state.joint_q.view(self.num_environments, -1)).sum(-1) > 0, torch.isnan(self.state.joint_qd.view(self.num_environments, -1)).sum(-1) > 0))
        inf_masks = torch.logical_or(torch.isinf(self.obs_buf).sum(-1) > 0, torch.logical_or(torch.isinf(self.state.joint_q.view(self.num_environments, -1)).sum(-1) > 0, torch.isinf(self.state.joint_qd.view(self.num_environments, -1)).sum(-1) > 0))
        invalid_value_masks = torch.logical_or((torch.abs(self.state.joint_q.view(self.num_environments, -1)) > 1e6).sum(-1) > 0,
                                                (torch.abs(self.state.joint_qd.view(self.num_environments, -1)) > 1e6).sum(-1) > 0)   
        invalid_masks = torch.logical_or(invalid_value_masks, torch.logical_or(nan_masks, inf_masks))

        self.reset_buf = torch.where(invalid_masks, torch.ones_like(self.reset_buf), self.reset_buf)

        self.rew_buf[invalid_masks] = 0.
    