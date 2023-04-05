import numpy as np
import torch
from typing import Optional, Union
from scipy.interpolate import CubicSpline
from warp.envs import WarpEnv
from shac.envs import DFlexEnv


class Policy:
    def __init__(
        self,
        num_actions: int,
        horizon: float = 0.5,
        dt: float = 1.0 / 60.0,
        max_steps: int = 512,
        params: Optional[np.ndarray] = None,
        policy_type: str = "zero",
        step: float = 0.0,
    ):
        self.num_actions = num_actions
        self.horizon = horizon
        self.step = step
        self.dt = dt
        self.max_steps = max_steps

        # Spline points
        steps = int(min(horizon / dt + 1, max_steps))
        self.timesteps = np.linspace(0, horizon, steps)
        self.params = params if params is not None else np.zeros((steps, num_actions))
        self.policy_type = policy_type

    def get_policy(self, params=None, noise=None):
        pol = None
        params = self.params if params is None else params
        if noise:
            params = params + noise * np.random.randn(*params.shape)
        if self.policy_type == "cubic":
            pol = CubicSpline(self.timesteps, params)
        elif self.policy_type == "zero":
            pol = lambda x: params[:, np.argwhere(x > self.timesteps)[-1].item()]
        else:
            assert self.policy_type == "linear"
            pol = lambda x: np.stack(
                [
                    np.interp(x, self.timesteps, params[:, i])
                    for i in range(self.num_actions)
                ]
            )
        return pol

    def action(self, t, params=None):
        params = self.params if params is None else params
        return self.get_policy()(t, params)


class Planner:
    """A sampling-based planner"""

    def __init__(
        self,
        policy: Policy,
        env: Union[WarpEnv, DFlexEnv],
        noise: float = 0.1,
    ):
        self.policy = policy
        self.noise = noise
        self.env = env
        self.num_trajectories = self.env.num_envs

    def optimize_policy(self):
        """Optimize the policy"""
        params = [self.policy.params] + [
            self.policy.params.copy()
            + self.noise * np.random.randn(*self.policy.params.shape)
            for _ in range(self.num_trajectories - 1)
        ]
        policies = [self.policy.get_policy(p) for p in params]

        rewards = self.rollout(policies)
        best_traj = torch.argmax(rewards).item()
        winner = params[best_traj]
        with torch.no_grad():
            self.clone_state(best_traj)
        self.policy.params = winner

    def clone_state(self, env_idx):
        """Clone the state of the environment"""
        if isinstance(self.env, DFlexEnv):
            joint_q = self.env.state.joint_q.view(self.num_trajectories, -1)
            joint_q[:] = joint_q[env_idx : env_idx + 1]
            joint_qd = self.env.state.joint_qd.view(self.num_trajectories, -1)
            joint_qd[:] = joint_qd[env_idx : env_idx + 1]
            joint_act = self.env.state.joint_act.view(self.num_trajectories, -1)
            joint_act[:] = joint_act[env_idx : env_idx + 1]
        else:
            ckpt = self.env.get_checkpoint()
            for k in ckpt:
                ckpt_v = ckpt[k].view(self.num_trajectories, -1)[env_idx : env_idx + 1]
                ckpt[k] = ckpt_v.repeat(self.num_trajectories, 1)
            self.env.load_checkpoint(ckpt)

    def rollout(self, policies=None, render=False):
        """Rollout the policy"""
        acc_rew = 0.0
        self.env.reset()
        self.policy.step = 0
        if policies is None:
            policies = [self.policy.get_policy()] * self.num_trajectories
        for t in range(self.policy.max_steps):
            action = torch.tensor(
                [policy(t) for policy in policies],
                device=self.env.device,
                dtype=torch.float32,
            )
            obs, reward, _, _ = self.env.step(action)
            if render:
                self.env.render()
            acc_rew += reward
            self.policy.step += 1

        return acc_rew
