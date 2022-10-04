# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property # and proprietary rights in and to this software, related documentation
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


def test_gradcheck_zero_ac(args):
    seeding()
    env_fn = getattr(envs, args.env)
    env_fn_kwargs = dict(
        num_envs=args.num_envs,
        device="cuda",
        render=args.render,
        seed=0,
        no_grad=False,
        stochastic_init=False  # True
    )
    if issubclass(env_fn, envs.DFlexEnv):
        env_fn_kwargs["MM_caching_frequency"] = 1

    env = env_fn(**env_fn_kwargs)
 
    def test_fn(actions, plot=False):
        ob_vec = []
        ob_vec.append(env.reset().detach().cpu().numpy().flatten())
        ret = 0
        for _ in range(1):
            obs, r, done, info = env.step(actions)
            print(obs)
            ob_vec.append(obs.detach().cpu().numpy().flatten())
            ret = r + ret * 0.999
        return r  
    actions = torch.tensor([[0.]], device='cuda', requires_grad=True)
    assert torch.autograd.gradcheck(test_fn, (actions,), eps=1, atol=1e-4)

def test_gradcheck_rand_ac(args):
    seeding()

    def test_fn(actions, plot=False):

        env_fn = getattr(envs, args.env)

        env_fn_kwargs = dict(
            num_envs=args.num_envs,
            device="cuda",
            render=args.render,
            seed=0,
            no_grad=False,
            stochastic_init=False  # True
        )
        if issubclass(env_fn, envs.DFlexEnv):
            env_fn_kwargs["MM_caching_frequency"] = 1

        env = env_fn(**env_fn_kwargs)
        ob_vec = []
        ob_vec.append(env.reset().detach().cpu().numpy().flatten())
        ret = 0
        for _ in range(1):
            obs, r, done, info = env.step(actions)
            print(obs)
            ob_vec.append(obs.detach().cpu().numpy().flatten())
            ret = r + ret * 0.999
        return r  
 
    num_actions = 1
    torch.manual_seed(123)
    actions = torch.rand(
        (args.num_envs, num_actions), device="cuda", requires_grad=True
    )
    for i in range(5):
        # reward = test_fn(actions, False)
        # if actions.grad:
        #     actions.grad.zero_()
        # reward.mean().backward()
        # print("reward:", reward)
        # print("actions:", actions)
        # print("actions.grad:", actions.grad)
        # if actions.grad:
        #    actions.grad.zero_()

        # check against finite differencing, eps,  atol rtol = 1e-6 
        assert torch.autograd.gradcheck(test_fn, (actions,), eps=1, atol=1e-4)
        # num_grad, anal_grad = check_grad(test_fn, actions)

        print("Finish Successfully")


def check_grad(fn, inputs, eps=1e-6, atol=1e-4, rtol=1e-6):
    if inputs.grad is not None:
        inputs.grad.zero_()
    out = fn(inputs)
    out.backward()
    analytical = inputs.grad.clone()
    x2, x1 = inputs + eps, inputs - eps
    numerical = (fn(x2) - fn(x1)) / (2 * eps)
    assert torch.allclose(numerical, analytical, rtol, atol), "numerical gradient was: {}, analytical was: {}".format(numerical, analytical)
    return (numerical, analytical)


def main(args):
    test_gradcheck_zero_ac(args)
    print("test_grad: zero action gradcheck passed")
    test_gradcheck_rand_ac(args)
    print("test_grad: rand action gradcheck passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPoleSwingUpWarpEnv")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--render", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
