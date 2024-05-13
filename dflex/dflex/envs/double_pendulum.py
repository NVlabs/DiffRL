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

import dflex.envs.load_utils as lu
import dflex.envs.torch_utils as tu


class DoublePendulumEnv(DFlexEnv):
    """ "
    The real state of the system is [theta1, theta2, theta1_dot, theta2_dot]
        where the thetas are the angle of the joints. Robot also known as Acrobot
    The observations are [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
    The actions are [theta2_torque] first joint is not controlled!
    """

    def __init__(
        self,
        render=False,
        device="cuda:0",
        num_envs=1024,
        episode_length=240,
        no_grad=True,
        stochastic_init=False,
        MM_caching_frequency=1,
        jacobian=False,
        start_state=[0.0, 0.0, 0.0, 0.0],
        logdir=None,
        nan_state_fix=False,
        jacobian_norm=None,
    ):
        num_obs = 6
        num_act =1

        super(DoublePendulumEnv, self).__init__(
            num_envs,
            num_obs,
            num_act,
            episode_length,
            MM_caching_frequency,
            no_grad,
            render,
            nan_state_fix,
            jacobian_norm,
            stochastic_init,
            jacobian,
            device,
        )

        self.early_termination=False
        self.start_state = np.array(start_state)
        assert self.start_state.shape[0] == 4, self.start_state

        self.init_sim()

        # action parameters
        self.action_strength = 1000.0

        # loss related
        self.pole_angle_penalty = 1.0
        self.pole_velocity_penalty = 0.1

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
        cartpole_filename = "doublependulum.urdf"
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


    def unscale_act(self, action):
        return action * self.action_strength

    def set_act(self, action):
        self.state.joint_act.view(self.num_envs, -1)[:, 1:2] = action

    
    def static_init_func(self, env_ids):
        joint_q = self.start_joint_q.view(self.num_envs, -1)[env_ids].clone()
        joint_qd = self.start_joint_qd.view(self.num_envs, -1)[env_ids].clone()
        return joint_q, joint_qd
    
    def stochastic_init_func(self, env_ids):
        """Method for computing stochastic init state"""
        joint_q, joint_qd = self.static_init_func(env_ids)
        joint_q += np.pi * (
            torch.rand(
                size=(len(env_ids), self.num_joint_q), device=self.device
            )
            - 0.5
        )

        joint_qd += 0.5 * (
            torch.rand(
                size=(len(env_ids), self.num_joint_qd), device=self.device
            )
            - 0.5
        )
        return joint_q, joint_qd

    def set_state_act(self, obs, act):
        self.state.joint_q.view(self.num_envs, -1)[:, 0] = obs[:, 0]
        theta = torch.atan2(obs.view(self.num_envs, -1)[:, 2], obs.view(self.num_envs, -1)[:, 3])
        theta = tu.normalize_angle(theta)
        self.state.joint_q.view(self.num_envs, -1)[:, 1] = theta
        self.state.joint_qd.view(self.num_envs, -1)[:, 0] = obs[:, 1]
        self.state.joint_qd.view(self.num_envs, -1)[:, 1] = obs[:, 4]
        self.state.joint_act.view(self.num_envs, -1)[:, 0] = act

    def observation_from_state(self, state):
        """Observation is [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]"""
        theta1 = state.joint_q.view(self.num_envs, -1)[:, 0:1]
        theta2 = state.joint_q.view(self.num_envs, -1)[:, 1:2]
        theta1_dot = state.joint_qd.view(self.num_envs, -1)[:, 0:1]
        theta2_dot = state.joint_qd.view(self.num_envs, -1)[:, 1:2]

        return torch.cat(
            [torch.cos(theta1), torch.sin(theta1), torch.cos(theta2), torch.sin(theta2), theta1_dot, theta2_dot], dim=-1
        )

    def calculate_reward(self, obs, act):
        # atan2(sin(theta), cos(theta))
        theta1 = torch.atan2(obs.view(self.num_envs, -1)[:, 1], obs.view(self.num_envs, -1)[:, 0])
        theta1 = tu.normalize_angle(theta1)
        theta2 = torch.atan2(obs.view(self.num_envs, -1)[:, 3], obs.view(self.num_envs, -1)[:, 2])
        theta2 = tu.normalize_angle(theta2)
        pole_angle_penalty = -torch.pow(theta1, 2.0) * self.pole_angle_penalty
        pole_angle_penalty += -torch.pow(theta2, 2.0) * self.pole_angle_penalty
        self.primal = pole_angle_penalty.clone()

        theta1_dot = obs.view(self.num_envs, -1)[:, 4]
        theta2_dot = obs.view(self.num_envs, -1)[:, 5]

        return (
            pole_angle_penalty
            - torch.pow(theta1_dot, 2.0) * self.pole_velocity_penalty
            - torch.pow(theta2_dot, 2.0) * self.pole_velocity_penalty
        )
