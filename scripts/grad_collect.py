import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from shac import envs
import numpy as np
import torch


@hydra.main(version_base="1.2", config_path="cfg", config_name="config.yaml")
def main(config: DictConfig):
    device = torch.device(config.general.device)
    torch.random.manual_seed(config.general.seed)

    # create environment
    env = instantiate(config.env.config)

    # create a random set of actions
    std = 0.5
    w = torch.normal(0.0, std, (env.num_envs, env.num_acts)).to(device)
    w[0] = w[0].zero_()
    fobgs = []
    zobgs = []

    for h in tqdm(range(1, env.episode_length)):
        print("h={:}".format(h))
        env.clear_grad()
        env.reset()

        ww = w.clone()
        ww.requires_grad_(True)

        loss = torch.zeros(env.num_envs).to(device)

        # apply first noisy action
        obs, rew, done, info = env.step(ww)
        loss += rew

        # let episode play out
        for t in range(1, h):
            obs, rew, done, info = env.step(torch.zeros_like(ww))
            loss += rew
            # NOTE: commented out code below is for the debugging of more efficient grad computation
            # make_dot(loss.sum(), show_attrs=True, show_saved=True).render("correct_graph")
            # loss.sum().backward(retain_graph=True)
            # print(ww.grad)
            # exit(1)

        loss.sum().backward()

        fobgs.append(ww.grad.cpu().numpy())

        # now get ZoBGs
        zobg = 1 / std**2 * (loss.unsqueeze(1) - loss[0]) * ww
        zobgs.append(zobg.detach().cpu().numpy())

    np.savez(
        "{:}_grads_{:}".format(env.__class__.__name__, env.episode_length),
        zobgs=np.array(zobgs),
        fobgs=np.array(fobgs),
    )


if __name__ == "__main__":
    main()
