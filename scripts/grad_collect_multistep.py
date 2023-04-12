import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from shac import envs
import numpy as np
import torch
from tqdm import tqdm
from torchviz import make_dot
from shac.envs import DFlexEnv
from shac.algorithms.shac import SHAC

# from torch.nn.utils import param
from torch.nn.utils import parameters_to_vector


@hydra.main(version_base="1.2", config_path="cfg", config_name="config.yaml")
def main(config: DictConfig):
    device = torch.device(config.general.device)
    torch.random.manual_seed(config.general.seed)

    # create environment
    env: DFlexEnv = instantiate(config.env)

    n = env.num_obs
    m = env.num_acts
    N = env.num_envs
    H = env.episode_length

    # Create actions
    # TODO theta should be only 1 parameter but I don't know how to get the grads
    #   with respect to different rollouts
    o = m  # parameter space TODO hardcoded
    th = torch.ones((N, o)).to(device)
    th.requires_grad_(True)

    # cartpole
    def policy(obs):
        # returns (N x m)
        # observation should be of shape (n_envs, n_obses)
        a = -th * obs[..., [1]]
        assert a.shape[-2:] == (N, m), a.shape
        return a

    # hopper
    def policy(obs):
        # returns (N x m)
        # observation should be of shape (n_envs, n_obses)
        a = -th * obs[..., [5, 5, 6]]
        assert a.shape[-2:] == (N, m), a.shape
        return a

    # create a random set of actions
    std = 0.5
    w = torch.normal(0.0, std, (H, N, m)).to(device)
    w[:, 0] = w[:, 0].zero_()
    fobgs = []
    zobgs = []
    zobgs_no_grad = []
    losses = []
    baseline = []

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

        # get losses
        loss.sum().backward()
        losses.append(loss.detach().cpu().numpy())
        baseline.append(loss[0].detach().cpu().numpy())

        # get First-order Batch Gradients (FoBGs)
        fobg = th.grad.cpu().numpy()
        assert fobg.shape == (N, o), fobg.shape
        fobgs.append(fobg)

        # get Zero-order Batch Gradients (ZoBGs)
        dpis = torch.stack(dpis)
        assert dpis.shape == (h, N, o), dpis.shape
        policy_grad = dpis * w[:h]
        assert policy_grad.shape == (h, N, o), policy_grad.shape
        policy_grad = policy_grad.sum(0)
        assert policy_grad.shape == (N, o), policy_grad.shape
        value = loss.unsqueeze(1) - loss[0]
        assert value.shape == (N, 1), value.shape
        zobg = 1 / std**2 * value * policy_grad
        assert zobg.shape == (N, o), zobg.shape
        zobgs.append(zobg.detach().cpu().numpy())

        # Now get ZoBGs without poliy gradients
        policy_grad = w[:h].sum(0)  # without policy gradients
        assert policy_grad.shape == (N, o), policy_grad.shape
        zobg_no_grad = 1 / std**2 * value * policy_grad
        zobgs_no_grad.append(zobg_no_grad.detach().cpu().numpy())

    # Save data
    filename = "{:}_grads_ms_{:}".format(
        env.__class__.__name__, config.env.episode_length
    )
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
        zobgs_no_grad=zobgs_no_grad,
        std=std,
        n=n,
        m=m,
    )


if __name__ == "__main__":
    main()
