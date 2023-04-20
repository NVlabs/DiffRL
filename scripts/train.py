import traceback
import hydra, os, wandb, yaml
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from shac.utils import custom_resolvers
from shac.algorithms.shac import SHAC
from shac.algorithms.shac2 import SHAC as SHAC2
from shac.utils.common import *


def create_wandb_run(wandb_cfg, job_config, run_id=None, run_wandb=False):
    try:
        job_id = HydraConfig().get().job.num
        override_dirname = HydraConfig().get().job.override_dirname
        name = f"{wandb_cfg.sweep_name_prefix}-{job_id}"
        notes = f"{override_dirname}"
    except:
        name, notes = None, None
    if run_wandb:
        return wandb.init(
            project=wandb_cfg.project,
            config=job_config,
            group=wandb_cfg.group,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            name=name,
            notes=notes,
            id=run_id,
            resume=run_id is not None,
        )


cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, "cfg")


@hydra.main(config_path="cfg", config_name="config.yaml")
def train(cfg: DictConfig):
    try:
        cfg_yaml = OmegaConf.to_yaml(cfg.alg)

        resume_model = cfg.resume_model
        if os.path.exists("exp_config.yaml"):
            old_config = yaml.load(open("exp_config.yaml", "r"))
            params, wandb_id = old_config["params"], old_config["wandb_id"]
            run = create_wandb_run(
                cfg.wandb, params, wandb_id, run_wandb=cfg.general.run_wandb
            )
            resume_model = "restore_checkpoint.zip"
            assert os.path.exists(
                resume_model
            ), "restore_checkpoint.zip does not exist!"
        else:
            defaults = HydraConfig.get().runtime.choices
            params = yaml.safe_load(cfg_yaml)
            params["defaults"] = {k: defaults[k] for k in ["alg"]}

            run = create_wandb_run(cfg.wandb, params, run_wandb=cfg.general.run_wandb)
            # wandb_id = run.id if run != None else None
            save_dict = dict(wandb_id=run.id if run != None else None, params=params)
            yaml.dump(save_dict, open("exp_config.yaml", "w"))
            print("Config:")
            print(cfg_yaml)

        if cfg.alg.name.startswith("shac"):
            alg_cls = SHAC if cfg.alg.name == "shac" else SHAC2
            cfg_train = yaml.safe_load(cfg_yaml)
            if cfg.general.play:
                cfg_train["params"]["config"]["num_actors"] = (
                    cfg_train["params"]["config"].get("player", {}).get("num_actors", 1)
                )
            if not cfg.general.no_time_stamp:
                cfg.general.logdir = os.path.join(cfg.general.logdir, get_time_stamp())

            cfg_train["params"]["general"] = yaml.safe_load(
                OmegaConf.to_yaml(cfg.general)
            )
            print(cfg_train["params"]["general"])
            if alg_cls == SHAC2:
                traj_optimizer = alg_cls(cfg)
            else:
                traj_optimizer = alg_cls(cfg_train)

            if not cfg.general.play:
                traj_optimizer.train()
            else:
                traj_optimizer.play(cfg_train)
        else:
            raise NotImplementedError
        wandb.finish()
    except:
        traceback.print_exc(file=open("exception.log", "w"))
        with open("exception.log", "r") as f:
            print(f.read())


if __name__ == "__main__":
    train()
