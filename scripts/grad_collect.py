import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from shac import envs
import numpy as np
import torch
from tqdm import tqdm
from torchviz import make_dot


@hydra.main(version_base="1.2", config_path="cfg", config_name="config.yaml")
def main(config: DictConfig):
    device = torch.device(config.general.device)
    torch.random.manual_seed(config.general.seed)

    # create environment
    env = instantiate(config.env)

    n = env.num_obs
    m = env.num_acts
    N = env.num_envs
    H = env.episode_length

    # create a random set of actions
    std = 0.5
    w = torch.normal(0.0, std, (N, m)).to(device)
    w[0] = w[0].zero_()
    fobgs = []
    zobgs = []
    losses = []
    baseline = []

    for h in tqdm(range(1, config.env.episode_length)):
        env.clear_grad()
        env.reset()

        ww = w.clone()
        ww.requires_grad_(True)
        loss = torch.zeros(config.env.num_envs).to(device)

        # apply first noisy action
        obs, rew, done, info = env.step(ww)
        loss += rew

        # let episode play out
        for t in range(1, h):
            obs, rew, done, info = env.step(torch.zeros_like(ww))
            loss += rew
            # NOTE: commented out code below is for the debugging of more efficient grad computation
            # make_dot(loss.sum(), show_attrs=True, show_saved=True).render("correct_graph")
            # loss.sum().backward()
            # print(ww.grad)
            # exit(1)

        loss.sum().backward()
        losses.append(loss.detach().cpu().numpy())
        baseline.append(loss[0].detach().cpu().numpy())

        # get First-order Batch Gradients (FoBGs)
        fobgs.append(ww.grad.cpu().numpy())

        # get Zero-order Batch Gradients (ZoBGs)
        zobg = 1 / std**2 * (loss.unsqueeze(1) - loss[0]) * ww
        zobgs.append(zobg.detach().cpu().numpy())

    filename = "{:}_grads_{:}".format(env.__class__.__name__, config.env.episode_length)
    if "warp" in config.env._target_:
        filename = "Warp" + filename
    filename = f"outputs/grads/{filename}"
    if hasattr(env, "start_state"):
        filename += "_" + str(env.start_state)
    print("Saving to", filename)
    np.savez(
        filename,
        zobgs=zobgs,
        fobgs=fobgs,
        losses=losses,
        baseline=baseline,
        std=std,
        n=n,
        m=m,
    )


if __name__ == "__main__":
    main()
