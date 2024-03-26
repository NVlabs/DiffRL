# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# gradient-based policy optimization by actor critic method
import os
import torch
from tensorboardX import SummaryWriter

from rl_games.algos_torch import torch_ext
from rl_games.common.algo_observer import AlgoObserver
from rl_games.common import env_configurations, vecenv


def parse_diff_env_kwargs(diff_env):
    env_kwargs = {}
    for key, value in diff_env.items():
        if key in [
            "name",
            "episode_length",
            "stochastic_env",
            "num_envs",
            "MM_caching_frequency",
            "no_grad",
            "render",
            "seed",
            "stochastic_init",
        ]:
            continue
        env_kwargs[key] = value
    print("parsed kwargs:", env_kwargs)
    return env_kwargs


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](
            **kwargs
        )

        self.full_state = {}

        self.rl_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.full_state["obs"] = self.env.reset(force_reset=True).to(self.rl_device)
        print(self.full_state["obs"].shape)

    def step(self, actions):
        self.full_state["obs"], reward, is_done, info = self.env.step(
            actions.to(self.env.device)
        )

        return (
            self.full_state["obs"].to(self.rl_device),
            reward.to(self.rl_device),
            is_done.to(self.rl_device),
            info,
        )

    def reset(self):
        self.full_state["obs"] = self.env.reset(force_reset=True)

        return self.full_state["obs"].to(self.rl_device)

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space
        info["agents"] = self.get_number_of_agents()

        print(info["action_space"], info["observation_space"])

        return info


class RLGPUEnvAlgoObserver(AlgoObserver):
    def after_init(self, algo):
        self.algo = algo
        self.score_keys = self.algo.config.get("score_keys", [])
        # dummy mean_scores to keep parent methods from breaking
        # TODO: find way to integrate better
        games_to_track = 100
        if hasattr(self.algo, "games_to_track"):
            games_to_track = self.algo.games_to_track
        device = self.algo.config.get("device", "cuda:0")

        self.mean_scores_map = {
            k + "_final": torch_ext.AverageMeter(1, games_to_track).to(device)
            for k in self.score_keys
        }

        if hasattr(self.algo, "writer"):
            self.writer = self.algo.writer
        else:
            summaries_dir = self.algo.summaries_dir
            os.makedirs(summaries_dir, exist_ok=True)
            self.writer = SummaryWriter(summaries_dir)

    def process_infos(self, infos, done_indices):
        super().process_infos(infos, done_indices)
        if isinstance(infos, dict):
            for k, v in filter(lambda kv: kv[0] in self.score_keys, infos.items()):
                final_v = v[done_indices]
                if final_v.shape[0] > 0:
                    self.mean_scores_map[f"{k}_final"].update(final_v)

    def after_clear_stats(self):
        for score_values in self.mean_scores_map.values():
            score_values.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        for score_key in self.score_keys:
            score_values = self.mean_scores_map[score_key + "_final"]
            if score_values.current_size > 0:
                mean_scores = score_values.get_mean()
                self.writer.add_scalar(f"scores/{score_key}/step", mean_scores, frame)
                self.writer.add_scalar(
                    f"scores/{score_key}/iter", mean_scores, epoch_num
                )
                self.writer.add_scalar(
                    f"scores/{score_key}/time", mean_scores, total_time
                )
