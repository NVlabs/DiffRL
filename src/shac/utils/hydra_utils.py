import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from typing import Dict

OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg in ["", None] else arg)

OmegaConf.register_new_resolver("eval", eval)


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret
