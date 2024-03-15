# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# gradient-based policy optimization by actor critic method
import sys, os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import argparse

from shac import envs
from shac.algorithms.shac import SHAC
import os
import sys
import yaml
import torch

import numpy as np
import copy

from shac.utils.common import *


def parse_arguments(description="Testing Args", custom_parameters=[]):
    parser = argparse.ArgumentParser()

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(
                        argument["name"],
                        type=argument["type"],
                        default=argument["default"],
                        help=help_str,
                    )
                else:
                    print("ERROR: default must be specified if using type")
            elif "action" in argument:
                parser.add_argument(
                    argument["name"], action=argument["action"], help=help_str
                )
        else:
            print()
            print(
                "ERROR: command line argument name, type/action must be defined, argument not added to parser"
            )
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args


def get_args():  # TODO: delve into the arguments
    custom_parameters = [
        {
            "name": "--test",
            "action": "store_true",
            "default": False,
            "help": "Run trained policy, no training",
        },
        {
            "name": "--cfg",
            "type": str,
            "default": "./cfg/shac/ant.yaml",
            "help": "Configuration file for training/playing",
        },
        {
            "name": "--play",
            "action": "store_true",
            "default": False,
            "help": "Run trained policy, the same as test",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "default": "Base",
            "help": "Path to the saved weights",
        },
        {"name": "--logdir", "type": str, "default": "logs/tmp/shac/"},
        {"name": "--save-interval", "type": int, "default": 0},
        {
            "name": "--no-time-stamp",
            "action": "store_true",
            "default": False,
            "help": "whether not add time stamp at the log path",
        },
        {"name": "--device", "type": str, "default": "cuda:0"},
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {
            "name": "--render",
            "action": "store_true",
            "default": False,
            "help": "whether generate rendering file.",
        },
        {
            "name": "--wandb",
            "action": "store_true",
            "default": False,
            "help": "whether to log with wandb.",
        },
        {
            "name": "--c-act",
            "help": "act penalty coefficient",
            "type": float,
            "default": 2e-3,
        },
        {
            "name": "--c-finger",
            "help": "finger position error coefficient",
            "type": float,
            "default": 1.0,
        },
        {
            "name": "--c-q",
            "help": "object orientation error coefficient",
            "type": float,
            "default": 10.0,
        },
        {
            "name": "--c-pos",
            "help": "object position error coefficient",
            "type": float,
            "default": 0.0,
        },
        {
            "name": "--c-ft",
            "help": "object goal force-torque error coefficient",
            "type": float,
            "default": 0.0,
        },
    ]

    # parse arguments
    args = parse_arguments(description="SHAC", custom_parameters=custom_parameters)

    return args


if __name__ == "__main__":
    import warp as wp

    wp.config.verify_cuda = True

    args = get_args()

    with open(args.cfg, "r") as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    if args.play or args.test:
        cfg_train["params"]["config"]["num_actors"] = (
            cfg_train["params"]["config"].get("player", {}).get("num_actors", 1)
        )

    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())

    args.device = torch.device(args.device)

    vargs = vars(args)

    cfg_train["params"]["general"] = {}
    for key in vargs.keys():
        cfg_train["params"]["general"][key] = vargs[key]

    if args.wandb:
        import wandb

        wandb.init(
            project="dmanip",
            config=cfg_train["params"],
            entity="krshna",
            sync_tensorboard=True,
            resume="allow",
        )

    traj_optimizer = SHAC(cfg_train)

    if args.train:
        traj_optimizer.train()
    else:
        traj_optimizer.play(cfg_train)
