# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property # and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from time import time

import torch
import functorch
import random

from shac import envs
from shac.utils.common import seeding

import argparse

import matplotlib.pyplot as plt
import numpy as np

from shac.utils import torch_utils as tu
from torch.autograd.functional import jacobian

from warp.envs import HopperEnv


def test_jac(args, num_steps):
    seeding()

    # env_fn = getattr(envs, args.env)

    # env_fn_kwargs = dict(
    #     num_envs=args.num_envs,
    #     device="cuda",
    #     render=args.render,
    #     seed=0,
    #     no_grad=False,
    #     stochastic_init=False,  # True
    # )
    # if issubclass(env_fn, envs.DFlexEnv):
    #     env_fn_kwargs["MM_caching_frequency"] = 1

    # env = env_fn(**env_fn_kwargs)
    env = HopperEnv(
        num_envs=args.num_envs, seed=0, no_grad=False, stochastic_init=False
    )
    ob_vec = []
    obs = env.reset()

    def f(inputs):
        # print("inputs", inputs.shape)
        # first set state
        # states = np.tile(states, (env.num_envs, len(states)))
        states = inputs[:, : env.num_obs]
        env.joint_q.view(env.num_envs, -1)[:, 0] = 0.0
        env.joint_q.view(env.num_envs, -1)[:, 1:] = states[:, :5]
        env.joint_qd.view(env.num_envs, -1)[:] = states[:, 5:]

        # compute and set action
        # actions = tu.to_torch(actions).view((env.num_envs, env.num_actions))
        # actions = inputs[:, env.num_obs :]
        # actions = torch.clip(actions, -1.0, 1.0)
        # unscaled_actions = actions * env.action_strength
        # env.state.joint_act.view(env.num_envs, -1)[:, 3:] = unscaled_actions
        env.assign_actions(tu.to_torch(actions))

        env.update()
        next_state = torch.cat(
            [
                env.joint_q.view(env.num_envs, -1)[:, 1:],
                env.joint_qd.view(env.num_envs, -1),
            ],
            dim=-1,
        )
        return next_state

    print("obs space:", env.num_obs)
    print("act space:", env.num_acts)
    states = tu.to_torch(np.random.randn(env.num_envs, env.num_obs))
    actions = tu.to_torch(np.random.uniform(-1, 1, (env.num_envs, env.num_acts)))
    print(states.shape)
    print(actions.shape)
    inputs = torch.cat((states, actions), dim=1)
    now = time()
    jac = jacobian(f, inputs)
    # this below is a faster function which sadly doesn't work
    # jac = functorch.vmap(functorch.jacrev(f))(inputs)
    print("took {:.2f}".format(time() - now))
    print("jacobian", jac.shape)

    # print(jac)

    # discard cross-batched data
    jac = torch.stack([jac[i, :, i] for i in range(env.num_envs)])
    print("jacobian", jac.shape)

    J = jac.detach().cpu().numpy()
    np.save("jac", J)

    for b in range(len(jac)):
        for i in range(jac.shape[1]):
            print(b, i, torch.norm(jac[b, i]))
        print(b, torch.norm(jac[b]))

    print("overall", torch.norm(jac))
    # print(jac)

    # print(env.model.J.shape)


def check_grad(fn, inputs, eps=1e-6, atol=1e-4, rtol=1e-6):
    if inputs.grad is not None:
        inputs.grad.zero_()
    out = fn(inputs)
    out.backward()
    analytical = inputs.grad.clone()
    x2, x1 = inputs + eps, inputs - eps
    numerical = (fn(x2) - fn(x1)) / (2 * eps)
    assert torch.allclose(
        numerical, analytical, rtol, atol
    ), "numerical gradient was: {}, analytical was: {}".format(numerical, analytical)
    return (numerical, analytical)


def main(args):
    test_jac(args, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPoleSwingUpEnv")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--render", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
