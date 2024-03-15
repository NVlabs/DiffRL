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

    fobgs = []
    zobgs = []
    losses = []
    baseline = []

    for h in tqdm(range(1, H + 1)):
        env = Bounce(num_envs=N, num_steps=h, profile=False, render=False)
        w = env.noise_[:, :2]

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

    # env.render_iter(0)  # render last interation

    directory = "outputs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "bounce_grads_{:}".format(H, clip)
    filename = f"{directory}/{filename}"
    print("Saving to", filename)
    np.savez(
        filename,
        h=np.arange(0, H),
        zobgs=zobgs,
        fobgs=fobgs,
        losses=losses,
        baseline=baseline,
        std=std,
        n=n,
        m=m,
    )

    # bounce.check_grad()
    # bounce.train_graph()


if __name__ == "__main__":
    main()
