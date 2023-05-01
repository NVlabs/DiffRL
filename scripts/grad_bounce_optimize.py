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
    N = 1024
    H = 40

    w = np.random.normal(0.0, std, (N, m))
    w[0] = 0.0  # for baseline
    ww = np.append(w, np.zeros((N, 1)), axis=1)

    env = Bounce(ww, num_envs=N, num_steps=H, profile=False, render=False)
    losses, trajectories = env.train(200)

    env = Bounce(ww, num_envs=N, num_steps=H, profile=False, render=False)
    losses_clip, trajectories_clip = env.train(200, clip=5.0)

    env = Bounce(ww, num_envs=N, num_steps=H, profile=False, render=False)
    losses_norm, trajectories_norm = env.train(200, norm=5.0)
    np.savez(
        "bounce_optimization.npz",
        losses=losses,
        trajectories=trajectories,
        losses_clip=losses_clip,
        trajectories_clip=trajectories_clip,
        losses_norm=losses_norm,
        trajectories_norm=trajectories_norm,
    )


if __name__ == "__main__":
    main()
