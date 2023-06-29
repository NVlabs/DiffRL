# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property # and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import argparse
import numpy as np
from time import time
import torch

from shac.utils import torch_utils as tu
from shac import envs
from shac.utils.common import seeding
from shac.utils.torch_utils import jacobian, jacobian2


EPS = 0.1
ATOL = 1e-4


def example_jac(args):
    seeding()

    # Create environment
    env_fn = getattr(envs, args.env)

    env_fn_kwargs = dict(
        num_envs=args.num_envs,
        device="cuda",
        render=False,
        seed=0,
        no_grad=False,
        stochastic_init=False,  # True
    )

    env = env_fn(**env_fn_kwargs)
    env.reset()

    def f(inputs):
        """Wrapper function for a single simulation step function"""
        # first set state
        states = inputs[:, : env.num_obs]
        env.state.joint_q.view(env.num_envs, -1)[:, 0] = 0.0
        env.state.joint_q.view(env.num_envs, -1)[:, 1:] = states[:, :5]
        env.state.joint_qd.view(env.num_envs, -1)[:] = states[:, 5:]

        # compute and set action
        actions = inputs[:, env.num_obs :]
        # actions = torch.clip(actions, -1.0, 1.0)
        # unscaled_actions = actions * env.action_strength
        env.state.joint_act.view(env.num_envs, -1)[:, 3:] = actions

        next_state = env.integrator.forward(
            env.model,
            env.state,
            env.sim_dt,
            env.sim_substeps,
            env.MM_caching_frequency,
            False,
        )
        next_state = torch.cat(
            [
                next_state.joint_q.view(env.num_envs, -1)[:, 1:],
                next_state.joint_qd.view(env.num_envs, -1),
            ],
            dim=-1,
        )
        return next_state

    print("obs space:", env.num_obs)
    print("act space:", env.num_acts)
    states = np.random.randn(env.num_envs, env.num_obs)
    actions = (
        np.random.uniform(-1, 1, (env.num_envs, env.num_acts)) * env.action_strength
    )
    inputs = tu.to_torch(np.concatenate((states, actions), axis=-1))
    inputs.requires_grad_(True)

    # Now compute jacobian
    now = time()
    jac = jacobian(f, inputs)
    total_time = time() - now
    print("took {:.2f}".format(total_time))
    print("jacobian shape", jac.shape)

    directory = "outputs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "jacs_{:}".format(args.env)
    filename = f"{directory}/{filename}"
    print("Saving to", filename)
    np.save(filename, jac.detach().cpu().numpy())

    for b in range(len(jac)):
        for i in range(jac.shape[1]):
            print(b, i, torch.norm(jac[b, i]))
        print(b, torch.norm(jac[b]))


def example_jac2(args):
    seeding()

    # Create environment
    env_fn = getattr(envs, args.env)

    env_fn_kwargs = dict(
        num_envs=args.num_envs,
        device="cuda",
        render=False,
        seed=0,
        no_grad=False,
        stochastic_init=False,  # True
    )

    env = env_fn(**env_fn_kwargs)
    env.reset()

    print("obs space:", env.num_obs)
    print("act space:", env.num_acts)
    states = np.random.randn(env.num_envs, env.num_obs)
    actions = (
        np.random.uniform(-1, 1, (env.num_envs, env.num_acts)) * env.action_strength
    )
    inputs = tu.to_torch(np.concatenate((states, actions), axis=-1))
    inputs.requires_grad_(True)

    # Set state
    states = inputs[:, : env.num_obs]
    env.state.joint_q.view(env.num_envs, -1)[:, 0] = 0.0
    env.state.joint_q.view(env.num_envs, -1)[:, 1:] = states[:, :5]
    env.state.joint_qd.view(env.num_envs, -1)[:] = states[:, 5:]

    # compute and set action
    actions = inputs[:, env.num_obs :]
    env.state.joint_act.view(env.num_envs, -1)[:, 3:] = actions

    next_state = env.integrator.forward(
        env.model,
        env.state,
        env.sim_dt,
        env.sim_substeps,
        env.MM_caching_frequency,
        False,
    )
    outputs = torch.cat(
        [
            next_state.joint_q.view(env.num_envs, -1)[:, 1:],
            next_state.joint_qd.view(env.num_envs, -1),
        ],
        dim=-1,
    )

    # Now compute jacobian
    now = time()
    jac = jacobian2(outputs, inputs)
    total_time = time() - now
    print("took {:.2f}".format(total_time))
    print("jacobian shape", jac.shape)

    directory = "outputs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "jacs2_{:}".format(args.env)
    filename = f"{directory}/{filename}"
    print("Saving to", filename)
    np.save(filename, jac.detach().cpu().numpy())

    for b in range(len(jac)):
        for i in range(jac.shape[1]):
            print(b, i, torch.norm(jac[b, i]))
        print(b, torch.norm(jac[b]))


def test_jac(args):
    seeding()

    # Create environment
    env_fn = getattr(envs, args.env)

    env_fn_kwargs = dict(
        num_envs=1,
        device="cuda",
        render=False,
        seed=0,
        no_grad=False,
        stochastic_init=False,  # True
    )

    env = env_fn(**env_fn_kwargs)
    env.reset()

    def f(inputs):
        """Wrapper function for a single simulation step function"""
        # first set state
        states = inputs[:, : env.num_obs]
        env.state.joint_q.view(env.num_envs, -1)[:, 0] = 0.0
        env.state.joint_q.view(env.num_envs, -1)[:, 1:] = states[:, :5]
        env.state.joint_qd.view(env.num_envs, -1)[:] = states[:, 5:]

        # compute and set action
        actions = inputs[:, env.num_obs :]
        actions = torch.clip(actions, -1.0, 1.0)
        unscaled_actions = actions * env.action_strength
        env.state.joint_act.view(env.num_envs, -1)[:, 3:] = unscaled_actions

        next_state = env.integrator.forward(
            env.model,
            env.state,
            env.sim_dt,
            env.sim_substeps,
            env.MM_caching_frequency,
            False,
        )
        next_state = torch.cat(
            [
                next_state.joint_q.view(env.num_envs, -1)[:, 1:],
                next_state.joint_qd.view(env.num_envs, -1),
            ],
            dim=-1,
        )
        return next_state.flatten()

    print("obs space:", env.num_obs)
    print("act space:", env.num_acts)
    states = tu.to_torch(np.random.randn(env.num_envs, env.num_obs))
    actions = tu.to_torch(np.random.uniform(-1, 1, (env.num_envs, env.num_acts)))
    inputs = torch.cat((states, actions), dim=1)
    inputs.requires_grad_(True)

    assert torch.autograd.gradcheck(f, (inputs,), eps=EPS, atol=ATOL)


def main(args):
    print("\nJacobian method 1")
    example_jac(args)

    print("\nJacobian method 2")
    example_jac2(args)

    if args.test:
        args.num_envs = 1
        test_jac(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HopperEnv")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--test", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
