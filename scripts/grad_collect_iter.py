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

    env = instantiate(config.env)

    # create a random set of actions
    std = 0.5
    w = torch.normal(0.0, std, (config.env.num_envs, env.num_acts)).to(device)
    w[0] = w[0].zero_()
    fobgs = []
    zobgs = []

    h = 200
    env.clear_grad()
    env.reset()

    ww = w.clone()
    ww.requires_grad_(True)
    loss = torch.zeros(config.env.num_envs).to(device)

    # apply first noisy action
    obs, rew, done, info = env.step(ww)
    loss += rew
    loss.sum().backward(retain_graph=True)

    # let episode play out
    for t in tqdm(range(1, h)):
        obs, rew, done, info = env.step(torch.zeros_like(ww))
        loss += rew
        ww.grad.zero_()  # do to make gradients correct
        # make_dot(loss.sum(), show_attrs=True, show_saved=True).render("bad_graph")
        loss.sum().backward(retain_graph=True)
        print(ww.grad)
        exit(1)

        fobgs.append(ww.grad.cpu().numpy())

        # now get ZoBGs
        zobg = 1 / std**2 * (loss.unsqueeze(1) - loss[0]) * ww
        zobgs.append(zobg.detach().cpu().numpy())

    np.savez(
        "{:}_grads2_{:}".format(env.__class__.__name__, h),
        zobgs=np.array(zobgs),
        fobgs=np.array(fobgs),
    )


if __name__ == "__main__":
    main()
