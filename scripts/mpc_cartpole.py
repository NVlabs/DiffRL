# coding: utf-8
from shac.algorithms.mpc import Policy, Planner
from shac.envs.cartpole_swing_up import CartPoleSwingUpEnv
import numpy as np
from tqdm import trange

EP_LEN = 240
env = CartPoleSwingUpEnv(num_envs=512, episode_length=EP_LEN)
eval_env = CartPoleSwingUpEnv(
    num_envs=1,
    episode_length=EP_LEN,
    render=True,
    stage_path="eval_mpc2",
)
p = Planner(Policy(env.num_acts, horizon=0.25), env)
rewards = []
for _ in trange(EP_LEN):
    p.optimize_policy()
    obs, rew, done, info = p.step(eval_env)
    rewards.append(rew.detach().cpu().numpy())

import matplotlib.pyplot as plt

plt.plot(rewards)
plt.savefig("rewards.png")
