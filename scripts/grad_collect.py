import hydra
from omegaconf import OmegaConf, DictConfig
from shac import envs
import numpy as np
import torch


@hydra.main(version_base="1.2", config_path="cfg", config_name="config.yaml")
def main(cfg: DictConfig):
    seed = 0
    ep_len = 200
    num_envs = 1024
    device = "cuda:0"

    torch.random(seed)

    env = envs.CartPoleSwingUpEnv(
        render=False,
        device=device,
        num_envs=num_envs,
        seed=0,
        episode_length=ep_len,
        no_grad=False,
        stochastic_init=False,
        MM_caching_frequency=4,
        early_termination=False,
        inverted_pendulum=True,
    )

    # create a random set of actions
    std = 0.5
    w = torch.normal(0.0, std, (num_envs, env.num_acts)).to(device)
    w[0] = w[0].zero_()
    fobgs = []
    zobgs = []

    for h in range(1, ep_len):
        print("h={:}".format(h))
        env.clear_grad()
        env.reset()

        ww = w.clone()
        ww.requires_grad_(True)

        loss = torch.zeros(num_envs).to(device)

        # apply first noisy action
        obs, rew, done, info = env.step(ww)
        loss += rew

        # let episode play out
        for t in range(1, h):
            obs, rew, done, info = env.step(torch.zeros_like(ww))
            loss += rew

        loss.sum().backward()

        fobgs.append(ww.grad.cpu().numpy())

        # now get ZoBGs
        zobg = 1 / std**2 * (loss.unsqueeze(1) - loss[0]) * ww
        zobgs.append(zobg.detach().cpu().numpy())

    np.savez(
        "{:}_grads_{:}".format(env.__class__.__name__, ep_len),
        zobgs=np.array(zobgs),
        fobgs=np.array(fobgs),
    )


if __name__ == "__main__":
    main()
