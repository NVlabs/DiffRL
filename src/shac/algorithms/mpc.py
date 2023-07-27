# only reset environment every 30 steps

import numpy as np
import torch
from typing import Optional, Union
from collections import namedtuple
from scipy.interpolate import CubicSpline
from warp.envs import WarpEnv
from shac.envs import DFlexEnv


CheckpointState = namedtuple("CheckpointState", ["joint_q", "joint_qd"])


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
        self.timesteps = np.arange(steps)  # np.linspace(0, horizon, steps)
        self.params = params if params is not None else np.zeros((steps, num_actions))
        self.policy_type = policy_type
        self._pi = None
        self.pi = self.get_policy(self.params)

    @property
    def pi(self):
        return self._pi

    @pi.setter
    def pi(self, new_pi):
        self._pi = new_pi
        self.step = 0  # reset step for nominal policy

    def get_policy(self, params=None, noise=None):
        pol = self._pi
        if pol is not None and params is None and noise is None:
            return pol  # early exit, return cached policy
        if params is None:
            params = self.params
        if noise is not None:
            params = params + noise * np.random.randn(*params.shape)

        if self.policy_type == "cubic":
            print(params.shape)
            pol = CubicSpline(np.arange(params.shape[0]), params, bc_type="natural")
        elif self.policy_type == "zero":
            pol = lambda x: params[
                min(self.max_steps - 1, np.searchsorted(self.timesteps[:-1], x))
            ]
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
        if params is None:
            return self._pi(t)
        return self.get_policy(params)(t)


# Add a reset function
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
        # with torch.no_grad():
        #     self.clone_state(best_traj)
        self.policy.params = winner
        self.policy.pi = policies[best_traj]

    def step(self, eval_env):
        """Step the environment, and all parallel envs to the same next state"""
        action = self.policy.action(self.policy.step)
        self.policy.step += self.policy.dt
        obs, reward, done, info = eval_env.step(
            torch.tensor(action, dtype=torch.float32, device=eval_env.device)
        )
        if isinstance(self.env, DFlexEnv):
            joint_q, joint_qd = eval_env.get_state()
            eval_ckpt = CheckpointState(joint_q=joint_q, joint_qd=joint_qd)
        else:
            eval_ckpt = eval_env.get_checkpoint()
        self.copy_eval_checkpoint(eval_ckpt)
        return obs, reward, done, info

    def clone_state(self, env_idx):
        """Clone the state of the environment"""
        if isinstance(self.env, DFlexEnv):
            joint_q = self.env.state.joint_q.view(self.num_trajectories, -1)[
                env_idx : env_idx + 1
            ]
            # joint_q[:] = joint_q[env_idx : env_idx + 1]
            joint_qd = self.env.state.joint_qd.view(self.num_trajectories, -1)[
                env_idx : env_idx + 1
            ]
            # joint_qd[:] = joint_qd[env_idx : env_idx + 1]
            eval_ckpt = CheckpointState(joint_q=joint_q, joint_qd=joint_qd)
        else:
            ckpt = self.env.get_checkpoint()
            eval_ckpt = {}
            for k, v in ckpt.items():
                if not k.endswith("buf") and k != "actions":
                    eval_ckpt[k] = v[env_idx : env_idx + 1]
        self.copy_eval_checkpoint(eval_ckpt)

    def copy_eval_checkpoint(self, eval_ckpt):
        if isinstance(eval_ckpt, CheckpointState):
            self.env.state.joint_q.view(self.num_trajectories, -1)[
                :
            ] = eval_ckpt.joint_q.view(1, -1)
            self.env.state.joint_qd.view(self.num_trajectories, -1)[
                :
            ] = eval_ckpt.joint_qd.view(1, -1)
        else:
            ckpt = self.env.get_checkpoint()
            for k in ckpt:
                ckpt_v = eval_ckpt[k].view(1, -1)
                ckpt[k] = ckpt_v.repeat(self.num_trajectories, 1)
            self.env.load_checkpoint(ckpt)

    def rollout(self, policies=None, render=False):
        """Rollout the policy"""
        acc_rew = 0.0
        self.policy.step = 0
        if policies is None:
            policies = [self.policy.get_policy()] * self.num_trajectories

        # rollout policy until horizon or end of episode reached
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

    def reset(self):
        self.env.reset()
