# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
from tqdm import tqdm

from bounce_env import Bounce

import warp as wp
import warp.sim
import warp.sim.render


def main():
    np.random.seed(0)

    std = 1e-1
    n = 2
    m = 2
    N = 128
    H = 40
    clip = 5.0

    w = np.random.normal(0.0, std, (N, m))
    w[0] = 0.0  # for baseline
    ww = np.append(w, np.zeros((N, 1)), axis=1)

    sweeps = [
        {
            "soft_contact_ke": 1e4,
            "soft_contact_kf": 1e0,
            "soft_contact_kd": 1e1,
            "soft_contact_mu": 0.9,
            "soft_contact_margin": 1e1,
        },
        {
            "soft_contact_ke": 2e4,
            "soft_contact_kf": 2e0,
            "soft_contact_kd": 2e1,
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
            "soft_contact_ke": 7e4,
            "soft_contact_kf": 7e0,
            "soft_contact_kd": 7e1,
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
        {
            "soft_contact_ke": 2e5,
            "soft_contact_kf": 2e1,
            "soft_contact_kd": 2e2,
            "soft_contact_mu": 0.9,
            "soft_contact_margin": 1e1,
        },
    ]

    results = []

    for i, params in enumerate(sweeps):
        print("Sweep", i)

        fobgs = []
        zobgs = []
        losses = []
        baseline = []

        for h in tqdm(range(1, H + 1)):
            env = Bounce(N, h, profile=False, render=False, **params)

            param = env.states[0].particle_qd

            tape = wp.Tape()
            with tape:
                loss = env.compute_loss()
                l = env.sum_loss()
            tape.backward(l)
            fobg = tape.gradients[param].numpy()
            fobg = fobg[:, :2]
            # gradient clipping by value
            # print(np.sum(np.abs(fobg) > clip))
            # fobg = np.clip(fobg, -clip, clip)

            # gradient clipping by norm
            # if np.any(fobg > clip):
            # fobg = clip * fobg / np.linalg.norm(fobg)
            tape.zero()
            loss = loss.numpy()

            losses.append(loss)
            baseline.append(loss[0])
            zobg = 1 / std**2 * (loss[..., None] - loss[0]) * w
            zobgs.append(zobg)
            fobgs.append(fobg)

        result = {
            "zobgs": np.array(zobgs),
            "fobgs": np.array(fobgs),
            "losses": np.array(losses),
            "baseline": np.array(baseline),
        }
        result.update(params)
        results.append(result)

    # env.render_iter(0)  # render last interation

    directory = "outputs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "bounce_grads_{:}_sim_sweep".format(H, clip)
    filename = f"{directory}/{filename}"
    print("Saving to", filename)
    np.save(filename, results)

    # bounce.check_grad()
    # bounce.train_graph()


if __name__ == "__main__":
    main()
