import hydra
from tqdm import trange
import shac.algorithms.mpc
import matplotlib.pyplot as plt
from shac.utils import custom_resolvers
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    env = instantiate(cfg.env.config)
    eval_env = instantiate(cfg.env.config, num_envs=1)
    policy = instantiate(cfg.alg.config.policy, num_actions=env.num_acts)
    planner = instantiate(cfg.alg.config.planner, env=env, policy=policy)
    rewards = run_planner(planner, eval_env)

    plt.plot(rewards)
    plt.savefig("rewards.png")


def run_planner(planner, eval_env):
    planner.reset()
    eval_env.reset()
    rewards = []
    for _ in trange(eval_env.episode_length):
        planner.optimize_policy()
        obs, rew, done, info = planner.step(eval_env)
        rewards.append(rew.detach().cpu().numpy())
        eval_env.render()  # ignored if render flag not passed
    return rewards


if __name__ == "__main__":
    main()
