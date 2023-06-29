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
from copy import deepcopy
from time import time
from functorch import combine_state_for_ensamble, vmap

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
    o = 4865

    # Load policy
    shac_path = "/home/ignat/git/SHAC/scripts/outputs/2023-04-08/18-59-28/logs/tmp/shac/04-08-2023-18-59-32/best_policy.pt"
    print("Loading policies")
    policies = []
    shac = SHAC(OmegaConf.to_container(config.alg, resolve=True), config.env)
    shac.load(shac_path)
    shac.actor.eval()
    for _ in range(N):
        new_actor = deepcopy(shac.actor)
        new_actor.eval()
        policies.append(new_actor)
    fmodel, params, buffers = combine_state_for_ensamble(policies)
    [p.requires_grad_() for p in params];
    print("Loaded policies")

    def policy(obs: torch.Tensor):
        # obs should be (NxO)
        # pre-process observations
        obs = torch.stack([obs for i in range(N)])
        actions = vmap(fmodel)(params, buffers, obs)
        # act should be [N, N, m]
        actions = torch.stack([actions[i, i] for i in range(N)])

        # post process observations
        # actions = []
        # for i in range(obs.shape[0]):
            # a = policies[i](obs[i], deterministic=True)
            # actions.append(a)
        # actions = torch.stack(actions)
        assert actions.shape == (N, m), actions.shape
        return actions

    parameters = [list(actor.parameters())[1:] for actor in policies]
    parameters = []
    for actor in policies:
        parameters.extend(list(actor.parameters())[1:])

    # create a random set of actions
    std = 0.5
    w = torch.normal(0.0, std, (H, N, m)).to(device)
    w[:, 0] = w[:, 0].zero_()
    fobgs = []
    zobgs = []
    # zobgs_no_grad = []

    for h in tqdm(range(1, H)):
        env.clear_grad()
        obs = env.reset()
        dpis = []
        loss = torch.zeros(N).to(device)

        # let episode play out
        for t in range(0, h):

            start = time()
            # Accumulate ZoBGs along the way
            # compute policy gradients along the way for FoBGs later
            grads = torch.autograd.grad(policy(obs.detach()).sum(), parameters)
            duration = time() - start
            print("ZoBG computation took {:.3f}s".format(duration))
            # reshape parameters into the shapes we want them in
            dpi = []
            for i in range(N):
                dpi_per_batch = grads[i*10: (i+1)*10]
                dpi_per_batch = [each.flatten() for each in dpi_per_batch]
                dpi_per_batch = torch.concat(dpi_per_batch)
                dpi.append(dpi_per_batch)
            dpi = torch.stack(dpi)
            assert dpi.shape == (N, o), dpi.shape
            dpis.append(dpi)
            duration = time() - start
            print("ZoBG accumulation took {:.3f}s".format(duration))

            # Rollout environment
            action = policy(obs) + w[t]
            obs, rew, done, info = env.step(action)
            loss += rew
            # NOTE: commented out code below is for the debugging of more efficient grad computation
            # make_dot(loss.sum(), show_attrs=True, show_saved=True).render("correct_graph")
            # loss.sum().backward()
            # print(ww.grad)
            # exit(1)

        # get first order gradients per environment
        start = time()
        loss.sum().backward()
        grads = []
        for actor in policies:
            grad_batch = []
            for name, param in actor.named_parameters():
                # print(name, param.shape)
                if name not in "logstd":
                    grad_batch.append(param.grad.flatten())
            grad_batch = torch.concat(grad_batch)
            grads.append(grad_batch)
        fobg = torch.stack(grads)
        assert fobg.shape == (N, o), dpis.shpae
        fobgs.append(fobg)
        duration = time() - start
        print("FoBG took {:.3f}s".format(duration))

        # now get ZoBGs
        start = time()
        dpis = torch.stack(dpis)
        assert dpis.shape == (h, N, o), dpis.shape
        policy_grad = dpis * w[:h]
        assert policy_grad.shape == (h, N, o), policy_grad.shape
        policy_grad = policy_grad.sum(0)
        assert policy_grad.shape == (N, o), policy_grad.shape
        baseline = loss[0]
        value = loss.unsqueeze(1) - baseline
        assert value.shape == (N, 1), value.shape
        zobg = 1 / std**2 * value * policy_grad
        assert zobg.shape == (N, o), zobg.shape
        zobgs.append(zobg.detach().cpu().numpy())
        duration = time() - start
        print("ZoBG took {:.3f}s",format(duration))

        # Now get ZoBGs without poliy gradients
        # policy_grad = w[:h].sum(0)  # without policy gradients
        # assert policy_grad.shape == (N, o), policy_grad.shape
        # zobg_no_grad = 1 / std**2 * value * policy_grad
        # zobgs_no_grad.append(zobg_no_grad.detach().cpu().numpy())

    np.savez(
        "{:}_grads_ms_{:}".format(env.__class__.__name__, config.env.episode_length),
        zobgs=zobgs,
        # zobgs_no_grad=zobgs_no_grad,
        fobgs=fobgs,
    )


if __name__ == "__main__":
    main()
