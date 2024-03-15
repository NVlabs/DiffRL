# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch


class CriticDataset:
    def __init__(self, batch_size, obs, target_values, shuffle=False, drop_last=False):
        self.obs = obs.view(-1, obs.shape[-1])
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size

        # filter nans
        not_nan_idx = (self.obs == self.obs).all(dim=-1).nonzero(as_tuple=False)

        # for debugging below
        # print("detected {:} nans".format(len(self.obs) - len(not_nan_idx)))
        # if len(self.obs) - len(not_nan_idx) > 0:
        #     print(self.obs.shape)
        #     print(not_nan_idx.shape)
        #     print(not_nan_idx)
        #     exit(1)

        self.obs = self.obs[not_nan_idx]
        self.target_values = self.target_values[not_nan_idx]

        if shuffle:
            self.shuffle()

        if drop_last:
            self.length = self.obs.shape[0] // self.batch_size
        else:
            self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1

    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index, :]
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.obs.shape[0])
        return {
            "obs": self.obs[start_idx:end_idx, :],
            "target_values": self.target_values[start_idx:end_idx],
        }


class QCriticDataset:
    def __init__(
        self, batch_size, obs, act, target_values, shuffle=False, drop_last=False
    ):
        self.obs = obs.view(-1, obs.shape[-1])
        self.act = act.view(-1, act.shape[-1])
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size

        if shuffle:
            self.shuffle()

        if drop_last:
            self.length = self.obs.shape[0] // self.batch_size
        else:
            self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1

    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index, :]
        self.act = self.act[index, :]
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.obs.shape[0])
        return {
            "obs": self.obs[start_idx:end_idx, :],
            "act": self.act[start_idx:end_idx, :],
            "target_values": self.target_values[start_idx:end_idx],
        }
