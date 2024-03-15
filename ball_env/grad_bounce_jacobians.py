# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
from tqdm import tqdm
from bounce_env import Bounce

import warp as wp
import warp.sim
import warp.sim.render


def main():
    np.random.seed(0)

    std = 1e-1
    n = 2
    m = 2
    N = 128
    H = 40

    env = Bounce(num_envs=N, num_steps=H, std=std)

    # get last model jacobians
    print("Computing dynamics jacobians")
    jacs = []

    for i in tqdm(range(env.sim_steps)):
        tape = wp.Tape()
        with tape:
            env.states[i].clear_forces()

            env.integrator.simulate(
                env.model, env.states[i], env.states[i + 1], env.sim_dt
            )

        # For each timestep compute the jacobian
        # due to the way backprop works, we have to compute it per output dimension
        jacobian = np.empty((N, 6, 6), dtype=np.float32)
        for out_idx in range(3):
            # env.state[i].particle_q should be [N, <wp.vec3>] shape
            # we want them to be
            # select which row of the Jacobian we want to compute
            select_index = np.zeros(3)
            select_index[out_idx] = 1.0
            e = wp.array(np.tile(select_index, N), dtype=wp.vec3)
            # pass input gradients to the output buffer to apply selection
            tape.backward(grads={env.states[i + 1].particle_q: e})
            dq_dq = tape.gradients[env.states[i].particle_q]
            dq_dqd = tape.gradients[env.states[i].particle_qd]
            tape.zero()
            tape.backward(grads={env.states[i + 1].particle_qd: e})
            dqd_dq = tape.gradients[env.states[i].particle_q]
            dqd_dqd = tape.gradients[env.states[i].particle_qd]
            jacobian[:, out_idx, :3] = dq_dq.numpy()
            jacobian[:, out_idx, 3:6] = dqd_dq.numpy()
            jacobian[:, out_idx + 3, :3] = dq_dqd.numpy()
            jacobian[:, out_idx + 3, 3:6] = dqd_dqd.numpy()
            tape.zero()

        jacs.append(jacobian)

    jacs = np.array(jacs)
    print("Jacobian has shape", jacs.shape)

    xy = []
    for state in env.states:
        xy.append(state.particle_q.numpy())

    directory = "outputs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "jacobians"
    filename = f"{directory}/{filename}"
    print("Saving to", filename)

    np.savez(filename, xy=xy, jacobians=jacs, std=std, n=n, m=m, H=H)


if __name__ == "__main__":
    main()
