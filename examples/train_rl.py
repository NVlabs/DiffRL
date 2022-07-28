# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner
from rl_games.algos_torch import torch_ext

import argparse

import envs
import os
import sys
import yaml

import numpy as np
import copy
import torch

from utils.common import *


def create_dflex_env(**kwargs):
    env_fn = getattr(envs, cfg_train["params"]["diff_env"]["name"])
    
    env = env_fn(num_envs=cfg_train["params"]["config"]["num_actors"], \
        render=args.render, seed=args.seed, \
        episode_length=cfg_train["params"]["diff_env"].get("episode_length", 1000), \
        no_grad=True, stochastic_init=cfg_train['params']['diff_env']['stochastic_env'], \
        MM_caching_frequency=cfg_train['params']['diff_env'].get('MM_caching_frequency', 1))

    print('num_envs = ', env.num_envs)
    print('num_actions = ', env.num_actions)
    print('num_obs = ', env.num_obs)

    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)

    return env


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

        self.full_state = {}

        self.rl_device = "cuda:0"

        self.full_state["obs"] = self.env.reset(force_reset=True).to(self.rl_device)
        print(self.full_state["obs"].shape)

    def step(self, actions):
        self.full_state["obs"], reward, is_done, info = self.env.step(actions.to(self.env.device))

        return self.full_state["obs"].to(self.rl_device), reward.to(self.rl_device), is_done.to(self.rl_device), info

    def reset(self):
        self.full_state["obs"] = self.env.reset(force_reset=True)

        return self.full_state["obs"].to(self.rl_device)

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space

        print(info['action_space'], info['observation_space'])

        return info

vecenv.register('DFLEX', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('dflex', {
    'env_creator': lambda **kwargs: create_dflex_env(**kwargs),
    'vecenv_type': 'DFLEX'})

def parse_arguments(description="Testing Args", custom_parameters=[]):
    parser = argparse.ArgumentParser()

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    print("ERROR: default must be specified if using type")
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)
        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
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

def get_args(): # TODO: delve into the arguments
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--num_envs", "type": int, "default": 0, "help": "Number of envirnments"},
        {"name": "--cfg", "type": str, "default": "./cfg/rl/ant.yaml",
            "help": "Configuration file for training/playing"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {"name": "--render", "action": "store_true", "default": False,
            "help": "whether generate rendering file."},
        {"name": "--logdir", "type": str, "default": "logs/tmp/rl/"},
        {"name": "--no-time-stamp", "action": "store_true", "default": False,
            "help": "whether not add time stamp at the log path"}]

    # parse arguments
    args = parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    return args



if __name__ == '__main__':

    args = get_args()

    with open(args.cfg, 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    if args.play or args.test:
        cfg_train["params"]["config"]["num_actors"] = cfg_train["params"]["config"].get("player", {}).get("num_actors", 1)

    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())

    if args.num_envs > 0:
        cfg_train["params"]["config"]["num_actors"] = args.num_envs
        
    vargs = vars(args)
    
    cfg_train["params"]["general"] = {}
    for key in vargs.keys():
        cfg_train["params"]["general"][key] = vargs[key]

    # save config
    if cfg_train['params']['general']['train']:
        log_dir = cfg_train["params"]["general"]["logdir"]
        os.makedirs(log_dir, exist_ok = True)
        # save config
        yaml.dump(cfg_train, open(os.path.join(log_dir, 'cfg.yaml'), 'w'))
        
    runner = Runner()
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)
