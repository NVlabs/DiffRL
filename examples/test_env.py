# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import time

import torch
import random

import envs
from utils.common import *

import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--env', type = str, default = 'AntEnv')
parser.add_argument('--num-envs', type = int, default = 64)
parser.add_argument('--render', default = False, action = 'store_true')

args = parser.parse_args()

seeding()

env_fn = getattr(envs, args.env)

env = env_fn(num_envs = args.num_envs, \
            device = 'cuda:0', \
            render = args.render, \
            seed = 0, \
            stochastic_init = True, \
            MM_caching_frequency = 16, \
            no_grad = True)

obs = env.reset()

num_actions = env.num_actions

t_start = time.time()

reward_episode = 0.
for i in range(1000):
    actions = torch.randn((args.num_envs, num_actions), device = 'cuda:0')
    obs, reward, done, info = env.step(actions)
    reward_episode += reward

t_end = time.time()

print('fps = ', 1000 * args.num_envs / (t_end - t_start))
print('mean reward = ', reward_episode.mean().detach().cpu().item())

print('Finish Successfully')

