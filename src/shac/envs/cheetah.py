# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
import sys

import torch

from .dflex_env import DFlexEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dflex as df

import numpy as np

np.set_printoptions(precision=5, linewidth=256, suppress=True)

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

from shac.utils import load_utils as lu
from shac.utils import torch_utils as tu


class CheetahEnv(DFlexEnv):
    def __init__(
        self,
        render=False,
        device="cuda:0",
        num_envs=4096,
        seed=0,
        episode_length=1000,
        no_grad=True,
        stochastic_init=False,
        MM_caching_frequency=16,
        early_termination=True,
        contact_termination=False,
        jacobians=False,
    ):
        num_obs = 17
        num_act = 6

        super(CheetahEnv, self).__init__(
            num_envs,
            num_obs,
            num_act,
            episode_length,
            MM_caching_frequency,
            seed,
            no_grad,
            render,
            device,
        )

        self.stochastic_init = stochastic_init
        self.early_termination = early_termination
        self.contact_termination = contact_termination
        self.jacobians = jacobians

        self.init_sim()

        # other parameters
        self.action_strength = 200.0
        self.action_penalty = -1e-1

        # -----------------------
        # set up Usd renderer
        if self.visualize:
            self.stage = Usd.Stage.CreateNew(
                "outputs/" + "Cheetah_" + str(self.num_envs) + ".usd"
            )

            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0 / 60.0
        self.sim_substeps = 16
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 9
        self.num_joint_qd = 9

        self.x_unit_tensor = tu.to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False
        ).repeat((self.num_envs, 1))

        self.start_rotation = torch.tensor(
            [0.0], device=self.device, requires_grad=False
        )

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()

        self.start_pos = []
        self.start_joint_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.start_joint_target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        start_height = -0.2

        asset_folder = os.path.join(os.path.dirname(__file__), "assets")
        for i in range(self.num_environments):
            link_start = len(self.builder.joint_type)

            lu.parse_mjcf(
                os.path.join(asset_folder, "half_cheetah.xml"),
                self.builder,
                density=1000.0,
                stiffness=0.0,
                damping=1.0,
                contact_ke=2e4,
                contact_kd=1e3,
                contact_kf=1e3,
                contact_mu=1.0,
                limit_ke=1e3,
                limit_kd=1e1,
                armature=0.1,
                radians=True,
                load_stiffness=True,
            )

            self.builder.joint_X_pj[link_start] = df.transform(
                (0.0, 1.0, 0.0),
                df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
            )

            # base transform
            self.start_pos.append([0.0, start_height])

            # set joint targets to rest pose in mjcf
            self.builder.joint_q[
                i * self.num_joint_q + 3 : i * self.num_joint_q + 9
            ] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.builder.joint_target[
                i * self.num_joint_q + 3 : i * self.num_joint_q + 9
            ] = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]

        self.start_pos = tu.to_torch(self.start_pos, device=self.device)
        self.start_joint_q = tu.to_torch(self.start_joint_q, device=self.device)
        self.start_joint_target = tu.to_torch(
            self.start_joint_target, device=self.device
        )

        # finalize model
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor(
            (0.0, -9.81, 0.0), dtype=torch.float32, device=self.device
        )

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()

        if self.model.ground:
            self.model.collide(self.state)

    def render(self, mode="human"):
        if self.visualize:
            self.render_time += self.dt
            self.renderer.update(self.state, self.render_time)

            render_interval = 1
            if self.num_frames == render_interval:
                try:
                    self.stage.Save()
                except:
                    print("USD save error")

                self.num_frames -= render_interval

    def step(self, actions, play=False):
        actions = actions.view((self.num_envs, self.num_actions))

        actions = torch.clip(actions, -1.0, 1.0)
        unscaled_actions = actions * self.action_strength
        self.actions = actions.clone()

        self.state.joint_act.view(self.num_envs, -1)[:, 3:] = unscaled_actions

        next_state = self.integrator.forward(
            self.model,
            self.state,
            self.sim_dt,
            self.sim_substeps,
            self.MM_caching_frequency,
        )

        # compute dynamics jacobians if requested
        if self.jacobians and not play:
            inputs = torch.cat((self.obs_buf.clone(), unscaled_actions.clone()), dim=1)
            inputs.requires_grad_(True)
            last_obs = inputs[:, : self.num_obs]
            act = inputs[:, self.num_obs :]
            self.setStateAct(last_obs, act)
            output = self.integrator.forward(
                self.model,
                self.state,
                self.sim_dt,
                self.sim_substeps,
                self.MM_caching_frequency,
                False,
            )
            outputs = torch.cat(
                [
                    output.joint_q.view(self.num_envs, -1)[:, 1:],
                    output.joint_qd.view(self.num_envs, -1),
                ],
                dim=-1,
            )
            # TODO why are there no jacobians for indices 11..17 ?
            jac = tu.jacobian2(outputs, inputs, max_out_dim=11)

        contact_changed = (
            next_state.contact_changed.clone() != self.state.contact_changed.clone()
        )
        num_contact_changed = (
            next_state.contact_changed.clone() - self.state.contact_changed.clone()
        )
        contact_changed = contact_changed.view(self.num_envs, -1).any(dim=1)
        num_contact_changed = num_contact_changed.view(self.num_envs, -1).sum(dim=1)
        self.state = next_state
        self.sim_time += self.sim_dt

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateReward()
        self.calculateObservations()

        # Reset environments if exseeded horizon
        # NOTE: this is truncation
        truncation = self.progress_buf > self.episode_length - 1

        # Reset environments if agent has ended in a bad state based on heuristics
        # NOTE: this is termination
        termination = torch.zeros_like(truncation)

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                "obs_before_reset": self.obs_buf_before_reset,
                "episode_end": self.termination_buf,
                "contact_changed": contact_changed,
                "num_contact_changed": num_contact_changed,
            }

            if self.jacobians and not play:
                self.extras.update({"jacobian": jac.cpu().numpy()})

        # reset all environments which have been terminated
        self.reset_buf = termination | truncation
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(
                    self.num_envs, dtype=torch.long, device=self.device
                )

        if env_ids is not None:
            # clone the state to avoid gradient error
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()

            # fixed start state
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:2] = self.start_pos[
                env_ids, :
            ].clone()
            self.state.joint_q.view(self.num_envs, -1)[
                env_ids, 2
            ] = self.start_rotation.clone()
            self.state.joint_q.view(self.num_envs, -1)[
                env_ids, 3:
            ] = self.start_joint_q.clone()
            self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.0

            # randomization
            if self.stochastic_init:
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:2] = (
                    self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:2]
                    + 0.1
                    * (torch.rand(size=(len(env_ids), 2), device=self.device) - 0.5)
                    * 2.0
                )
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 2] = (
                    torch.rand(len(env_ids), device=self.device) - 0.5
                ) * 0.2
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:] = (
                    self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:]
                    + 0.1
                    * (
                        torch.rand(
                            size=(len(env_ids), self.num_joint_q - 3),
                            device=self.device,
                        )
                        - 0.5
                    )
                    * 2.0
                )
                self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.5 * (
                    torch.rand(
                        size=(len(env_ids), self.num_joint_qd), device=self.device
                    )
                    - 0.5
                )

            # clear action
            self.actions = self.actions.clone()
            self.actions[env_ids, :] = torch.zeros(
                (len(env_ids), self.num_actions), device=self.device, dtype=torch.float
            )

            self.progress_buf[env_ids] = 0

            self.calculateObservations()

        return self.obs_buf

    def clear_grad(self, checkpoint=None):
        """cut off the gradient from the current state to previous states"""

        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}
                checkpoint["joint_q"] = self.state.joint_q.clone()
                checkpoint["joint_qd"] = self.state.joint_qd.clone()
                checkpoint["actions"] = self.actions.clone()
                checkpoint["progress_buf"] = self.progress_buf.clone()

            self.state = self.model.state()
            self.state.joint_q = checkpoint["joint_q"]
            self.state.joint_qd = checkpoint["joint_qd"]
            self.actions = checkpoint["actions"]
            self.progress_buf = checkpoint["progress_buf"]

    """
    This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and it returns the observation vectors
    """

    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()

        return self.obs_buf

    def get_checkpoint(self):
        checkpoint = {}
        checkpoint["joint_q"] = self.state.joint_q.clone()
        checkpoint["joint_qd"] = self.state.joint_qd.clone()
        checkpoint["actions"] = self.actions.clone()
        checkpoint["progress_buf"] = self.progress_buf.clone()

        return checkpoint

    def setStateAct(self, obs, act):
        # self.state.joint_q.view(self.num_envs, -1)[:, 0:2] = TODO Don't need
        self.state.joint_q.view(self.num_envs, -1)[:, 1:] = obs[:, :8]
        self.state.joint_qd.view(self.num_envs, -1)[:, :] = obs[:, 8:]
        self.state.joint_act.view(self.num_envs, -1)[:, 3:] = act

    def calculateObservations(self):
        self.obs_buf = torch.cat(
            [
                self.state.joint_q.view(self.num_envs, -1)[:, 1:],
                self.state.joint_qd.view(self.num_envs, -1),
            ],
            dim=-1,
        )

    def calculateReward(self):
        progress_reward = self.obs_buf[:, 8]

        self.rew_buf = (
            progress_reward + torch.sum(self.actions**2, dim=-1) * self.action_penalty
        )
