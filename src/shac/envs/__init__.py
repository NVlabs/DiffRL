# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .ant import AntEnv
from .cartpole_swing_up import CartPoleSwingUpEnv
from .cheetah import CheetahEnv
from .dflex_env import DFlexEnv
from .hopper import HopperEnv
from .humanoid import HumanoidEnv
from .snu_humanoid import SNUHumanoidEnv

from warp.envs.cartpole_swing_up import CartPoleSwingUpEnv as CartPoleSwingUpWarpEnv
from warp.envs.hopper import HopperEnv as HopperWarpEnv
from warp.envs.ant import AntEnv as AntWarpEnv

from warp.envs.obj_env import ObjectTask
from warp.envs.hand_env import HandObjectTask
from warp.envs.repose_task import ReposeTask
from warp.envs.articulate_task import ArticulateTask

# dmanip envs
try:
    from dmanip.envs import WarpEnv, ClawWarpEnv, AllegroWarpEnv
except ImportError:
    print("dmanip not found, skipping dmanip envs")
    pass
