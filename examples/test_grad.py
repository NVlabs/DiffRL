# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import time

import torch
import random

import envs
from utils.common import seeding

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main(args):
    seeding()

    env_fn = getattr(envs, args.env)

    env_fn_kwargs = dict(
        num_envs=args.num_envs,
        device="cuda",
        render=args.render,
        seed=0,
        no_grad=False,
    )
    if issubclass(env_fn, envs.DFlexEnv):
        env_fn_kwargs["MM_caching_frequency"] = 16

    env = env_fn(**env_fn_kwargs)

    def test_fn(actions, plot=False):
        ob_vec = []
        env.initialize_trajectory()
        ob_vec.append(env.reset().detach().cpu().numpy().flatten())
        for _ in range(10):
            obs, reward, done, info = env.step(actions)
            ob_vec.append(obs.detach().cpu().numpy().flatten())
        if plot:
            ob_vec = np.array(ob_vec)
            fig, axs = plt.subplots(nrows=1, ncols=ob_vec.shape[1])
            for i in range(ob_vec.shape[1]):
                axs[i].plot(ob_vec[:, i])
                axs[i].grid()
            plt.show()
        return reward

    num_actions = env.num_actions
    torch.manual_seed(123)
    actions = torch.randn(
        (args.num_envs, num_actions), device="cuda", requires_grad=True
    )
    reward = test_fn(actions, False)
    reward.mean().backward()
    print(reward)
    print(actions.grad)

    # check against finite differencing
    assert torch.autograd.gradcheck(test_fn, (actions,), eps=1e-6, atol=1e-4)

    # t_start = time.time()

    # reward_episode = 0.0
    # for i in range(1000):
    #     actions = torch.randn((args.num_envs, num_actions), device="cuda")
    #     obs, reward, done, info = env.step(actions)
    #     reward_episode += reward

    # t_end = time.time()

    # print("fps = ", 1000 * args.num_envs / (t_end - t_start))
    # print("mean reward = ", reward_episode.mean().detach().cpu().item())

    print("Finish Successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPoleSwingUpWarpEnv")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--render", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
