import hydra
from typing import Union
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
    th = torch.tensor([1.0]).to(device)
    th.requires_grad_(True)
    o = th.shape  # parameter space

    def policy(obs):
        # returns (N x m)
        # observation should be of shape (n_envs, n_obses)
        a = -th * obs[:, 1].view((-1, 1))
        assert a.shape == (N, m)
        return a

    # create a random set of actions
    std = 0.5
    w = torch.normal(0.0, std, (H, N, m)).to(device)
    w[:, 0] = w[:, 0].zero_()
    fobgs = []
    zobgs = []
    zobgs_no_grad = []

    for h in tqdm(range(1, H)):
        env.clear_grad()
        obs_hist = torch.empty((h, N, n)).to(device)
        obs = env.reset()
        obs_hist[0] = obs.clone()
        loss = torch.zeros(N).to(device)

        # let episode play out
        for t in range(0, h):
            obs, rew, done, info = env.step(policy(obs) + w[t])
            if t + 1 < h:
                obs_hist[t + 1] = obs.clone()
            loss += rew
            # NOTE: commented out code below is for the debugging of more efficient grad computation
            # make_dot(loss.sum(), show_attrs=True, show_saved=True).render("correct_graph")
            # loss.sum().backward()
            # print(ww.grad)
            # exit(1)

        # get FoBGs per environment
        # This here is a more efficient attempt at computing batch gradients which still doesn't work
        (grads,) = torch.autograd.grad(
            loss.sum(), (th,), (torch.ones_like(loss),), is_grads_batched=True
        )
        print(grads.shape)
        print(grads)
        exit(1)
        fobg = []
        for i in range(len(loss)):
            (grad,) = torch.autograd.grad(loss[i], th, retain_graph=True)
            fobg.append(grad.cpu().numpy())

        fobg = np.stack(fobg)
        assert fobg.shape == (N, 1), fobg.shape
        fobgs.append(fobg)

        # now get ZoBGs
        policy_grad = -obs_hist[:, :, [1]] * w[:h]
        assert policy_grad.shape == (h, N, m), policy_grad.shape
        policy_grad = policy_grad.sum(0)
        assert policy_grad.shape == (N, m), policy_grad.shape
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
        fobgs=fobgs,
    )


if __name__ == "__main__":
    main()
