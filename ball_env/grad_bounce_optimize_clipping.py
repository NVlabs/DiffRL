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

    env = Bounce(std=std, num_envs=N, num_steps=H, profile=False, render=False)
    z_losses, z_trajectories, z_norms = env.train(200, zero_order=True)

    env = Bounce(std=std, num_envs=N, num_steps=H, profile=False, render=False)
    losses, trajectories, norms = env.train(200)

    env = Bounce(std=std, num_envs=N, num_steps=H, profile=False, render=False)
    losses_clip, trajectories_clip, norms_clip = env.train(200, clip=1.0)

    env = Bounce(std=std, num_envs=N, num_steps=H, profile=False, render=False)
    losses_norm, trajectories_norm, norms_norm = env.train(200, norm=1.0)
    np.savez(
        "bounce_optimization_edge.npz",
        z_losses=z_losses,
        z_trajectories=z_trajectories,
        z_norms=z_norms,
        losses=losses,
        trajectories=trajectories,
        norms=norms,
        losses_clip=losses_clip,
        trajectories_clip=trajectories_clip,
        norms_clip=norms_clip,
        losses_norm=losses_norm,
        trajectories_norm=trajectories_norm,
        norms_norm=norms_norm,
    )


if __name__ == "__main__":
    main()
