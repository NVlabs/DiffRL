# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .ant import AntEnv
from .cartpole_swing_up import CartPoleSwingUpEnv
from .double_pendulum import DoublePendulumEnv
from .cheetah import CheetahEnv
from .dflex_env import DFlexEnv
from .hopper import HopperEnv
from .humanoid import HumanoidEnv
from .snu_humanoid import SNUHumanoidEnv

try:
    from warp.envs.cartpole_swing_up import CartPoleSwingUpEnv as CartPoleSwingUpWarpEnv
except:
    print("WARN: Couldn't import warp. Is it installed?")

try:
    from .cartpole_swing_up_warp import CartPoleSwingUpWarpEnv
except:
    print("WARN: Couldn't import CartPoleSwingUpWarpEnv. Is warp it installed?")

try:
    from dmanip.envs import WarpEnv, ClawWarpEnv, AllegroWarpEnv
except:
    print("WARN: Couldn't import dmanip envs. Is it installed?")
