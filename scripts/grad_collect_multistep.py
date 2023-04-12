from typing import Union
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from shac import envs
import numpy as np
import torch
from tqdm import tqdm
from torchviz import make_dot
from shac.envs import DFlexEnv, WarpEnv


@hydra.main(version_base="1.2", config_path="cfg", config_name="config.yaml")
def main(config: DictConfig):
    device = torch.device(config.general.device)
    torch.random.manual_seed(config.general.seed)

    # create environment
    env: Union[DFlexEnv, WarpEnv] = instantiate(config.env.config)

    n = env.num_obs
    m = env.num_acts
    N = env.num_envs
    H = env.episode_length

    # Create actions
    # TODO theta should be only 1 parameter but I don't know how to get the grads
    #   with respect to different rollouts
    th = torch.ones((N, 1)).to(device)
    th.requires_grad_(True)
    o = th.shape  # parameter space

    def policy(obs):
        # returns (N x m)
        # observation should be of shape (n_envs, n_obses)
        a = -th * obs[..., [1]]
        assert a.shape[-2:] == (N, m)
        return a

    # create a random set of actions
    std = 0.5
    w = torch.normal(0.0, std, (H, N, m)).to(device)
    w[:, 0] = w[:, 0].zero_()
    fobgs = []
    zobgs = []
    zobgs_no_grad = []
    zobgs_analytical = []

    for h in tqdm(range(1, H)):
        env.clear_grad()
        obs = env.reset()
        dpis = []
        loss = torch.zeros(N).to(device)

        # let episode play out
        for t in range(0, h):
            # compute policy gradients along the way for FoBGs later
            (dpi,) = torch.autograd.grad(policy(obs.detach()).sum(), th)
            dpis.append(dpi)
            action = policy(obs) + w[t]
            obs, rew, done, info = env.step(action)
            loss += rew
            # NOTE: commented out code below is for the debugging of more efficient grad computation
            # make_dot(loss.sum(), show_attrs=True, show_saved=True).render("correct_graph")
            # loss.sum().backward()
            # print(ww.grad)
            # exit(1)

        # get first order gradients per environment
        loss.sum().backward()
        fobg = th.grad.cpu().numpy()
        assert fobg.shape == (N, 1), fobg.shape
        fobgs.append(fobg)

        # now get ZoBGs
        dpis = torch.stack(dpis)
        assert dpis.shape == (h, N, 1), dpis.shape
        policy_grad = dpis * w[:h]
        assert policy_grad.shape == (h, N, 1), policy_grad.shape
        # NOTE: policy_grad should be with shape (h, N, o) but I couldn't make it work for the time being
        policy_grad = policy_grad.sum(0)
        assert policy_grad.shape == (N, 1), policy_grad.shape
        # NOTE: policy_grad should be with shape (N, o) but I couldn't make it work for the time being
        baseline = loss[0]
        value = loss.unsqueeze(1) - baseline
        assert value.shape == (N, 1), value.shape
        zobg = 1 / std**2 * value * policy_grad
        assert zobg.shape == (N, 1), zobg.shape
        zobgs.append(zobg.detach().cpu().numpy())

        # Now get ZoBGs without poliy gradients
        policy_grad = w[:h].sum(0)  # without policy gradients
        assert policy_grad.shape == (N, m), policy_grad.shape
        zobg_no_grad = 1 / std**2 * value * policy_grad
        zobgs_no_grad.append(zobg_no_grad.detach().cpu().numpy())

    np.savez(
        "{:}_grads_ms_{:}".format(
            env.__class__.__name__, config.env.config.episode_length
        ),
        zobgs=zobgs,
        zobgs_no_grad=zobgs_no_grad,
        zobgs_analytical=zobgs_analytical,
        fobgs=fobgs,
    )


if __name__ == "__main__":
    main()
