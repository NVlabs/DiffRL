# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import numpy as np
from tqdm import tqdm

from bounce_env import Bounce

import warp as wp
import warp.sim
import warp.sim.render


@hydra.main(version_base="1.2", config_path="cfg", config_name="config.yaml")
def main(config: DictConfig):
    np.random.seed(config.general.seed)

    std = 1e-1
    n = 2
    m = 2
    N = 128
    H = 40

    w = np.random.normal(0.0, std, (N, m))
    w[0] = 0.0  # for baseline
    ww = np.append(w, np.zeros((N, 1)), axis=1)

    fobgs = []
    zobgs = []
    losses = []
    baseline = []

    for h in tqdm(range(H, H + 1)):
        env = Bounce(ww, num_envs=N, num_steps=h, profile=False, render=False)

        param = env.states[0].particle_qd

        tape = wp.Tape()
        with tape:
            loss = env.compute_loss()
            l = env.sum_loss()
        tape.backward(l)
        fobg = tape.gradients[param].numpy()
        fobg = fobg[:, :2]
        tape.zero()
        loss = loss.numpy()

        losses.append(loss)
        baseline.append(loss[0])
        zobg = 1 / std**2 * (loss[..., None] - loss[0]) * w
        zobgs.append(zobg)
        fobgs.append(fobg)

    # env.render_iter(0)  # render last interation

    filename = "bounce_grads_{:}".format(H)
    filename = f"outputs/grads/{filename}"
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
