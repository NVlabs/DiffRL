import hydra
import warp as wp

# wp.config.mode = "debug"
# wp.config.print_launches = True
# wp.config.verify_cuda = True

from tqdm import trange
from shac.algorithms.mpc import Policy, Planner
import matplotlib.pyplot as plt

from shac.utils import hydra_utils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="cfg", config_name="mpc_config")
def main(cfg: DictConfig):
    env = instantiate(cfg.env.config)
    cfg.general.num_envs = 1
    cfg.general.render = True
    eval_env = instantiate(cfg.env.config)

    policy: Policy = instantiate(cfg.alg.config.policy, num_actions=env.num_acts, max_steps=env.episode_length)
    planner: Planner = instantiate(cfg.alg.config.planner, env=env, policy=policy)
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
