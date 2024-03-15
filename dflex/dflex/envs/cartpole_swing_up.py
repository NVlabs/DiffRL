# TODO need tom finish this bad boy

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

import dflex.envs.load_utils as lu
import dflex.envs.torch_utils as tu


class CartPoleSwingUpEnv(DFlexEnv):
    """ "
    The real state of the system is [x, x_dot, theta, theta_dot]
        where x is position on slide and theta is angle of the pendulum.
        theta=0 is pendulum pointing upwards
    The observations are [x, x_dot, sin(theta), cos(theta), theta_dot]
    The actions are [x_ddot]
    """

    def __init__(
        self,
        render=False,
        device="cuda:0",
        num_envs=1024,
        seed=0,
        episode_length=240,
        no_grad=True,
        stochastic_init=False,
        MM_caching_frequency=1,
        early_termination=False,
        jacobian=False,
        start_state=[0.0, 0.0, 0.0, 0.0],
        logdir=None,
        nan_state_fix=False,
        jacobian_norm=None,
        reset_all=False,
    ):
        num_obs = 5
        num_act = 1

        super(CartPoleSwingUpEnv, self).__init__(
            num_envs,
            num_obs,
            num_act,
            episode_length,
            MM_caching_frequency,
            seed,
            no_grad,
            render,
            nan_state_fix,
            jacobian_norm,
            reset_all,
            stochastic_init,
            jacobian,
            device,
        )

        self.start_state = np.array(start_state)
        assert self.start_state.shape[0] == 4, self.start_state

        self.stochastic_init = stochastic_init
        self.early_termination = early_termination
        self.init_sim()

        # action parameters
        self.action_strength = 1000.0

        # loss related
        self.pole_angle_penalty = 1.0
        self.pole_velocity_penalty = 0.1

        self.cart_position_penalty = 0.05
        self.cart_velocity_penalty = 0.1

        self.cart_action_penalty = 0.0

        self.setup_visualizer(logdir)


    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0 / 60.0
        self.sim_substeps = 4
        self.sim_dt = self.dt

        if self.visualize:
            self.env_dist = 1.0
        else:
            self.env_dist = 0.0

        self.num_joint_q = 2
        self.num_joint_qd = 2

        asset_folder = os.path.join(os.path.dirname(__file__), "assets")
        cartpole_filename = "cartpole.urdf"
        for i in range(self.num_environments):
            lu.urdf_load(
                self.builder,
                os.path.join(asset_folder, cartpole_filename),
                df.transform(
                    (0.0, 2.5, 0.0 + self.env_dist * i),
                    df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
                ),
                floating=False,
                armature=0.1,
                stiffness=0.0,
                damping=0.0,
                shape_ke=1e4,
                shape_kd=1e4,
                shape_kf=1e2,
                shape_mu=0.5,
                limit_ke=1e2,
                limit_kd=1.0,
            )
            self.builder.joint_q[i * self.num_joint_q] = self.start_state[0]
            self.builder.joint_q[i * self.num_joint_q + 1] = self.start_state[1]
            self.builder.joint_qd[i * self.num_joint_q] = self.start_state[2]
            self.builder.joint_qd[i * self.num_joint_q + 1] = self.start_state[3]

        self.model = self.builder.finalize(self.device)
        self.model.ground = False
        self.model.gravity = torch.tensor(
            (0.0, -9.81, 0.0), dtype=torch.float, device=self.device
        )

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()
        self.start_joint_q = self.state.joint_q.clone()
        self.start_joint_qd = self.state.joint_qd.clone()

    # Now inherited from class above
    # def render(self, mode="human"):
    #     if self.visualize:
    #         self.render_time += self.dt
    #         self.renderer.update(self.state, self.render_time)
    #         if self.num_frames == 40:
    #             try:
    #                 self.stage.Save()
    #             except:
    #                 print("USD save error")
    #             self.num_frames -= 40

    def step(self, actions):
        with df.ScopedTimer("simulate", active=False, detailed=False):
            actions = actions.view((self.num_envs, self.num_actions))

            actions = torch.clip(actions, -1.0, 1.0)
            self.actions = actions

            self.state.joint_act.view(self.num_envs, -1)[:, 0:1] = (
                actions * self.action_strength
            )

            self.state = self.integrator.forward(
                self.model,
                self.state,
                self.sim_dt,
                self.sim_substeps,
                self.MM_caching_frequency,
            )
            self.sim_time += self.sim_dt

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                "obs_before_reset": self.obs_buf_before_reset,
                "episode_end": self.termination_buf,
            }

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # self.obs_buf_before_reset = self.obs_buf.clone()

        with df.ScopedTimer("reset", active=False, detailed=False):
            if len(env_ids) > 0:
                self.reset(env_ids)

        with df.ScopedTimer("render", active=False, detailed=False):
            self.render()

        # self.extras = {'obs_before_reset': self.obs_buf_before_reset}

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(
                    self.num_envs, dtype=torch.long, device=self.device
                )

        if env_ids is not None:
            # fixed start state
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()
            self.state.joint_q.view(self.num_envs, -1)[
                env_ids, :
            ] = self.start_joint_q.view(-1, self.num_joint_q)[env_ids, :].clone()
            self.state.joint_qd.view(self.num_envs, -1)[
                env_ids, :
            ] = self.start_joint_qd.view(-1, self.num_joint_qd)[env_ids, :].clone()

            if self.stochastic_init:
                self.state.joint_q.view(self.num_envs, -1)[
                    env_ids, :
                ] = self.state.joint_q.view(self.num_envs, -1)[env_ids, :] + np.pi * (
                    torch.rand(
                        size=(len(env_ids), self.num_joint_q), device=self.device
                    )
                    - 0.5
                )

                self.state.joint_qd.view(self.num_envs, -1)[
                    env_ids, :
                ] = self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] + 0.5 * (
                    torch.rand(
                        size=(len(env_ids), self.num_joint_qd), device=self.device
                    )
                    - 0.5
                )

            self.progress_buf[env_ids] = 0

            self.calculateObservations()

        return self.obs_buf

    """
    cut off the gradient from the current state to previous states
    """

    def clear_grad(self):
        with torch.no_grad():  # TODO: check with Miles
            current_joint_q = self.state.joint_q.clone()
            current_joint_qd = self.state.joint_qd.clone()
            current_joint_act = self.state.joint_act.clone()
            self.state = self.model.state()
            self.state.joint_q = current_joint_q
            self.state.joint_qd = current_joint_qd
            self.state.joint_act = current_joint_act

    """
    This function starts collecting a new trajectory from the current states but cut off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and return the observation vectors
    """

    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def calculateObservations(self):
        x = self.state.joint_q.view(self.num_envs, -1)[:, 0:1]
        theta = self.state.joint_q.view(self.num_envs, -1)[:, 1:]
        xdot = self.state.joint_qd.view(self.num_envs, -1)[:, 0:1]
        theta_dot = self.state.joint_qd.view(self.num_envs, -1)[:, 1:]

        # observations: [x, xdot, sin(theta), cos(theta), theta_dot]
        self.obs_buf = torch.cat(
            [x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1
        )

    def calculateReward(self):
        x = self.state.joint_q.view(self.num_envs, -1)[:, 0]
        theta = tu.normalize_angle(self.state.joint_q.view(self.num_envs, -1)[:, 1])
        xdot = self.state.joint_qd.view(self.num_envs, -1)[:, 0]
        theta_dot = self.state.joint_qd.view(self.num_envs, -1)[:, 1]
        pole_angle_penalty = -torch.pow(theta, 2.0) * self.pole_angle_penalty

        self.rew_buf = (
            pole_angle_penalty
            - torch.pow(theta_dot, 2.0) * self.pole_velocity_penalty
            - torch.pow(x, 2.0) * self.cart_position_penalty
            - torch.pow(xdot, 2.0) * self.cart_velocity_penalty
            - torch.sum(self.actions**2, dim=-1) * self.cart_action_penalty
        )

        # reset agents
        self.reset_buf = torch.where(
            self.progress_buf > self.episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
