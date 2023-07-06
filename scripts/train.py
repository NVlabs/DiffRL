import traceback
import hydra, os, wandb, yaml
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from shac.utils import hydra_utils
from shac.algorithms.shac import SHAC
from shac.algorithms.shac2 import SHAC as SHAC2
from shac.algorithms.ahac import AHAC
from shac.utils.common import *
from shac.utils.rlgames_utils import (
    RLGPUEnvAlgoObserver,
    RLGPUEnv,
    parse_diff_env_kwargs,
)
from shac import envs
from gym import wrappers
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv


def register_envs(cfg_train):
    def create_dflex_env(**kwargs):
        env_fn = getattr(envs, cfg_train["params"]["diff_env"]["name"])

        env = env_fn(
            num_envs=cfg_train["params"]["config"]["num_actors"],
            render=cfg_train["params"]["general"]["render"],
            seed=cfg_train["params"]["general"]["seed"],
            episode_length=cfg_train["params"]["diff_env"].get("episode_length", 1000),
            no_grad=True,
            stochastic_init=cfg_train["params"]["diff_env"]["stochastic_init"],
            MM_caching_frequency=cfg_train["params"]["diff_env"].get(
                "MM_caching_frequency", 1
            ),
        )

        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)

        frames = kwargs.pop("frames", 1)
        if frames > 1:
            env = wrappers.FrameStack(env, frames, False)

        return env

    def create_warp_env(**kwargs):
        env_fn = getattr(envs, cfg_train["params"]["diff_env"]["name"])
        env_kwargs = parse_diff_env_kwargs(cfg_train["params"]["diff_env"])

        env = env_fn(
            num_envs=cfg_train["params"]["config"]["num_actors"],
            render=cfg_train["params"]["general"]["render"],
            seed=cfg_train["params"]["general"]["seed"],
            episode_length=cfg_train["params"]["diff_env"].get("episode_length", 1000),
            no_grad=True,
            stochastic_init=cfg_train["params"]["diff_env"]["stochastic_init"],
            **env_kwargs,
        )

        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)

        frames = kwargs.pop("frames", 1)
        if frames > 1:
            env = wrappers.FrameStack(env, frames, False)

        return env

    vecenv.register(
        "DFLEX",
        lambda config_name, num_actors, **kwargs: RLGPUEnv(
            config_name, num_actors, **kwargs
        ),
    )
    env_configurations.register(
        "dflex",
        {
            "env_creator": lambda **kwargs: create_dflex_env(**kwargs),
            "vecenv_type": "DFLEX",
        },
    )

    vecenv.register(
        "WARP",
        lambda config_name, num_actors, **kwargs: RLGPUEnv(
            config_name, num_actors, **kwargs
        ),
    )
    env_configurations.register(
        "warp",
        {
            "env_creator": lambda **kwargs: create_warp_env(**kwargs),
            "vecenv_type": "WARP",
        },
    )


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
        cfg_full = OmegaConf.to_container(cfg, resolve=True)
        cfg_yaml = yaml.dump(cfg_full["alg"])

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

        if "shac" in cfg.alg.name:
            if cfg.alg.name == "shac":
                cfg_train = cfg_full["alg"]
                if cfg.general.play:
                    cfg_train["params"]["config"]["num_actors"] = (
                        cfg_train["params"]["config"]
                        .get("player", {})
                        .get("num_actors", 1)
                    )
                if not cfg.general.no_time_stamp:
                    cfg.general.logdir = os.path.join(
                        cfg.general.logdir, get_time_stamp()
                    )

                cfg_train["params"]["general"] = cfg_full["general"]
                cfg_train["params"]["diff_env"] = cfg_full["env"]["config"]
                env_name = cfg_train["params"]["diff_env"].pop("_target_")
                cfg_train["params"]["diff_env"]["name"] = env_name.split(".")[-1]
                print(cfg_train["params"]["general"])
                traj_optimizer = SHAC(cfg_train)
            elif cfg.alg.name == "shac2":
                cfg_train = cfg_full["alg"]
                if cfg.general.play:
                    cfg_train["params"]["config"]["num_actors"] = (
                        cfg_train["params"]["config"]
                        .get("player", {})
                        .get("num_actors", 1)
                    )
                if not cfg.general.no_time_stamp:
                    cfg.general.logdir = os.path.join(
                        cfg.general.logdir, get_time_stamp()
                    )

                cfg_train["params"]["general"] = cfg_full["general"]
                cfg_train["params"]["diff_env"] = cfg_full["env"]["config"]
                env_name = cfg_train["params"]["diff_env"].pop("_target_")
                cfg_train["params"]["diff_env"]["name"] = env_name.split(".")[-1]
                print(cfg_train["params"]["general"])
                traj_optimizer = SHAC2(cfg_train)
            if not cfg.general.play:
                traj_optimizer.train()
            else:
                traj_optimizer.play(cfg_train)
            wandb.finish()
        elif cfg.alg.name == "ppo":
            cfg_train = cfg_full["alg"]
            cfg_train["params"]["general"] = cfg_full["general"]
            env_name = cfg_train["params"]["config"]["env_name"]
            cfg_train["params"]["diff_env"] = cfg_full["env"]["config"]
            if env_name.split("_")[0] == "df":
                cfg_train["params"]["config"]["env_name"] = "dflex"
            elif env_name.split("_")[0] == "warp":
                cfg_train["params"]["config"]["env_name"] = "warp"
            env_name = cfg_train["params"]["diff_env"].pop("_target_")
            cfg_train["params"]["diff_env"]["name"] = env_name.split(".")[-1]

            # save config
            if cfg_train["params"]["general"]["train"]:
                log_dir = cfg_train["params"]["general"]["logdir"]
                os.makedirs(log_dir, exist_ok=True)
                # save config
                yaml.dump(cfg_train, open(os.path.join(log_dir, "cfg.yaml"), "w"))

            # register envs
            register_envs(cfg_train)

            # add observer to score keys
            if cfg_train["params"]["config"].get("score_keys"):
                algo_observer = RLGPUEnvAlgoObserver()
            else:
                algo_observer = None
            runner = Runner(algo_observer)
            runner.load(cfg_train)
            runner.reset()
            runner.run(cfg_train["params"]["general"])
    except:
        traceback.print_exc(file=open("exception.log", "w"))
        with open("exception.log", "r") as f:
            print(f.read())


if __name__ == "__main__":
    train()
