# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import argparse
from concurrent import futures

configs = {
    "Ant": "ant.yaml",
    "CartPole": "cartpole_swing_up.yaml",
    "Hopper": "hopper.yaml",
    "Cheetah": "cheetah.yaml",
    "Humanoid": "humanoid.yaml",
    "SNUHumanoid": "snu_humanoid.yaml",
    "CartPoleWarp": "cartpole_swing_up_warp.yaml",
    "ClawWarp": "claw.yaml",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default="Ant",
    choices=[
        "Ant",
        "CartPole",
        "Hopper",
        "Cheetah",
        "Humanoid",
        "SNUHumanoid",
        "CartPoleWarp",
        "ClawWarp",
    ],
)
parser.add_argument(
    "--algo", type=str, default="shac", choices=["shac", "ppo", "sac", "bptt"]
)
parser.add_argument("--num-seeds", type=int, default=5)
parser.add_argument("--save-dir", type=str, default="./logs/")

args = parser.parse_args()

""" generate seeds """
seeds = []
for i in range(args.num_seeds):
    seeds.append(i * 10)

""" generate commands """
commands = []
for i in range(len(seeds)):
    seed = seeds[i]
    save_dir = os.path.join(args.save_dir, args.env, args.algo, str(seed))
    config_path = os.path.join("./cfg", args.algo, configs[args.env])

    if args.algo == "shac":
        script_name = "train_shac.py"
    elif args.algo == "ppo" or args.algo == "sac":
        script_name = "train_rl.py"
    elif args.algo == "bptt":
        script_name = "train_bptt.py"
    else:
        raise NotImplementedError

    cmd = (
        "python {} "
        "--cfg {} "
        "--seed {} "
        "--wandb "
        "--no-time-stamp "
        "--logdir {} ".format(script_name, config_path, seed, save_dir)
        # "--no-time-stamp ".format(script_name, config_path, seed, save_dir)
    )

    commands.append(cmd)

with futures.ThreadPoolExecutor(max_workers=10) as executor:
    jobs = [executor.submit(os.system, command) for command in commands]
    results = [job.result() for job in jobs]
    if any(results):
        print(f"jobs: {[i for i, x in enumerate(results) if x]} failed, check the logs")
    print(sum(job.done() for job in jobs), "jobs done")
