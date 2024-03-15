# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os
import numpy as np

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import time

import torch
import random

from shac import envs
from shac.utils.common import seeding

import argparse


def main(args):
    seeding()

    env_fn = getattr(envs, args.env)

    env_fn_kwargs = dict(
        num_envs=args.num_envs,
        device="cuda",
        render=args.render,
        seed=0,
        stochastic_init=False,
        no_grad=True,
    )
    if issubclass(env_fn, envs.DFlexEnv):
        env_fn_kwargs["MM_caching_frequency"] = 1

    env = env_fn(**env_fn_kwargs)
    # sets seed
    torch.manual_seed(123)

    obs = env.reset()
    print(obs[:1])

    num_actions = env.num_actions

    t_start = time.time()

    reward_episode = 0.0
    observations = np.load("data/observations.npy")
    # acts = np.load('data/acts.npy')
    for i in range(1000):
        actions = torch.rand((args.num_envs, num_actions), device="cuda")
        # act = actions.cpu().numpy()
        # assert np.isclose(acts[i], act).all(), f"expected {acts[i]} got {act}"
        obs, reward, done, info = env.step(actions)
        # assert obs.requires_grad, "obs should require grad"
        o = obs[:1].detach().cpu().numpy()
        # assert np.isclose(o, observations[i]).all(), f"expected {observations[i]} got {o}"
        # observations.append(obs[:1].detach().cpu().numpy())
        if i % 40 == 0:
            print(obs[:1])
        reward_episode += reward

    t_end = time.time()
    # np.save('data/observations.npy', np.stack(observations))

    print("fps = ", 1000 * args.num_envs / (t_end - t_start))
    print("mean reward = ", reward_episode.mean().detach().cpu().item())

    print("Finish Successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPoleSwingUpWarpEnv")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--render", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
