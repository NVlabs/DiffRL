import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from shac import envs
import numpy as np
import torch
import warp as wp
from tqdm import tqdm
from torchviz import make_dot


@hydra.main(version_base="1.2", config_path="cfg", config_name="config.yaml")
def main(config: DictConfig):
    device = torch.device(config.general.device)

    torch.random.manual_seed(config.general.seed)

    env = instantiate(config.env.config)

    # create a random set of actions
    std = 0.5
    n = env.num_obs
    m = env.num_acts
    w = torch.normal(0.0, std, (env.num_envs, env.num_acts)).to(device)
    w[0] = w[0].zero_()
    fobgs = []
    zobgs = []
    losses = []
    baseline = []

    h = env.episode_length
    env.clear_grad()
    env.reset()

    ww = w.clone()
    ww.requires_grad_(True)
    loss = torch.zeros(env.num_envs).to(device)

    # apply first noisy action
    obs, rew, done, info = env.step(ww)
    loss += rew
    loss.sum().backward(retain_graph=True)

    # let episode play out
    for t in tqdm(range(1, h)):
        profiler = {}
        obs, rew, done, info = env.step(torch.zeros_like(ww), profiler)
        loss += rew
        ww.grad.zero_()  # do to make gradients correct
        # make_dot(loss.sum(), show_attrs=True, show_saved=True).render("bad_graph")
        loss.sum().backward(retain_graph=True)
        losses.append(loss.detach().cpu().numpy())
        baseline.append(loss[0].detach().cpu().numpy())
        # print(ww.grad)
        # exit(1)

        fobgs.append(ww.grad.cpu().numpy())

        # now get ZoBGs
        zobg = 1 / std**2 * (loss.unsqueeze(1) - loss[0]) * ww
        zobgs.append(zobg.detach().cpu().numpy())

    filename = "{:}_grads2_{:}".format(env.__class__.__name__, h)
    print("saving to {}".format(filename))
    np.savez(
        filename,
        zobgs=np.array(zobgs),
        fobgs=np.array(fobgs),
        losses=np.array(losses),
        baseline=np.array(baseline),
        std=std,
        n=n,
        m=m,
    )


if __name__ == "__main__":
    main()
