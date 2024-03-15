# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import argparse
import subprocess
from concurrent import futures


def run_subprocess(command):
    command = command + f"& echo started job $!"
    with subprocess.Popen(
        command,
        shell=True,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        with open(f"job-{proc.pid}.out", "wb") as out_log:
            with open(f"job-{proc.pid}.err", "wb") as err_log:
                out_log.write(proc.stdout.read())
                err_log.write(proc.stderr.read())


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

    for command in commands:
        run_subprocess(command)
