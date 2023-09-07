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

try:
    from warp.envs.utils import hydra_resolvers
    from warp.envs.obj_env import ObjectTask
    from warp.envs.hopper import HopperEnv as HopperWarpEnv
    from warp.envs.hand_env import HandObjectTask
    from warp.envs.repose_task import ReposeTask
    from warp.envs.articulate_task import ArticulateTask
except ImportError as e:
    print("ERROR: warp envs not found, skipping warp envs")
    print(e)
    pass

# dmanip envs
try:
    from dmanip.envs import WarpEnv, ClawWarpEnv, AllegroWarpEnv
except ImportError as e:
    print("ERROR: dmanip not found, skipping dmanip envs")
    print(e)
    pass
