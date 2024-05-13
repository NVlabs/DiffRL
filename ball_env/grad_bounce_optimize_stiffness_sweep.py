# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
from tqdm import tqdm
import torch
from time import time

from bounce_env import Bounce

import warp as wp
import warp.sim
import warp.sim.render


def main():
    np.random.seed(0)

    std = 1e-1
    n = 2
    m = 2
    ndim = 30
    N = 1024
    H = 40

    # iterate over different parametarisations
    sweeps = [
        {
            "soft_contact_ke": 1e3,
            "soft_contact_kf": 1e0,
            "soft_contact_kd": 0.5,
            "soft_contact_mu": 0.9,
            "soft_contact_margin": 1e1,
        },
        {
            "soft_contact_ke": 3e4,
            "soft_contact_kf": 3e0,
            "soft_contact_kd": 3e1,
            "soft_contact_mu": 0.9,
            "soft_contact_margin": 1e1,
        },
        {
            "soft_contact_ke": 5e4,
            "soft_contact_kf": 5e0,
            "soft_contact_kd": 5e1,
            "soft_contact_mu": 0.9,
            "soft_contact_margin": 1e1,
        },
        {
            "soft_contact_ke": 1e5,
            "soft_contact_kf": 1e1,
            "soft_contact_kd": 1e2,
            "soft_contact_mu": 0.9,
            "soft_contact_margin": 1e1,
        },
    ]

    results = []

    print("Collecting landscape")
    for i, params in tqdm(enumerate(sweeps)):
        print("Sweep {:}/{:}".format(i + 1, len(sweeps)))

        temp_N = ndim * ndim
        env = Bounce(num_envs=temp_N, num_steps=H, std=0.0, render=True, **params)

        # plot landscape
        print("Collecting landscape")
        xx = np.linspace(0, 15, ndim)
        yy = np.linspace(-10, 5, ndim)
        vels = []
        for i, x in enumerate(xx):
            for j, y in enumerate(yy):
                vels.append((x, y, 0.0))
        vels = np.array(vels)
        env.reset(start_vel=vels)
        landscape = env.compute_loss().numpy()
        landscape_traj = env.trajectory()
        env.render_iter(0)

        print("First-order optimisation")
        env = Bounce(num_envs=N, num_steps=H, std=std, **params)
        losses, trajectories, _ = env.train(200)

        print("Zero-order optimisation")
        env = Bounce(num_envs=N, num_steps=H, std=std, **params)
        zero_losses, zero_trajectories, _ = env.train(500, zero_order=True)

        result = {
            "landscape": landscape,
            "landscape_trajectories": landscape_traj,
            "losses": losses,
            "trajectories": trajectories,
            "zero_losses": zero_losses,
            "zero_trajectories": zero_trajectories,
        }
        result.update(params)
        results.append(result)

    directory = "outputs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = f"landscapes_{std:.1f}"
    filename = f"{directory}/{filename}"
    print("Saving to", filename)

    # TODO make this saving more space efficient
    np.save(filename, results)


if __name__ == "__main__":
    main()
