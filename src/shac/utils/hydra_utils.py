import numpy as np
from omegaconf import OmegaConf, DictConfig
from typing import Dict

OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg in ["", None] else arg
)

OmegaConf.register_new_resolver("eval", eval)


def resolve_child(default, node, arg):
    """
    Attempts to get a child node parameter `arg` from `node`. If not
        present, the return `default`
    """
    if arg in node:
        return node[arg]
    else:
        return default


OmegaConf.register_new_resolver("resolve_child", resolve_child)


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


from dmanip.utils.common import ActionType, HandType, ObjectType, GoalType, RewardType
from dmanip.envs.environment import RenderMode

OmegaConf.register_new_resolver("object", lambda x: ObjectType[x.upper()])
OmegaConf.register_new_resolver("hand", lambda x: HandType[x.upper()])
OmegaConf.register_new_resolver("action", lambda x: ActionType[x.upper()])
OmegaConf.register_new_resolver("goal", lambda x: GoalType[x.upper()])
OmegaConf.register_new_resolver("reward", lambda x: RewardType[x.upper()])
OmegaConf.register_new_resolver("render", lambda x: RenderMode[x.upper()])
