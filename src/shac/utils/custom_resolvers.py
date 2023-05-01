import numpy as np
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg == "" else arg)

OmegaConf.register_new_resolver("eval", eval)
