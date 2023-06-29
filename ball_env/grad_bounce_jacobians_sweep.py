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

    w = np.random.normal(0.0, std, (N, m))
    w[0] = 0.0  # for baseline
    w = np.append(w, np.zeros((N, 1)), axis=1)

    # iterate over different parametarisations
    sweeps = [
        {
            "soft_contact_ke": 1e4,
            "soft_contact_kf": 1e0,
            "soft_contact_kd": 1e1,
            "soft_contact_mu": 0.9,
            "soft_contact_margin": 1e1,
        },
        {
            "soft_contact_ke": 3e4,
            "soft_contact_kf": 3e0,
            "soft_contact_kd": 3e1,
            "soft_contact_mu": 0.9,
            "soft_contact_margin": 1e1,
        },
        {
            "soft_contact_ke": 5e4,
            "soft_contact_kf": 5e0,
            "soft_contact_kd": 5e1,
            "soft_contact_mu": 0.9,
            "soft_contact_margin": 1e1,
        },
        {
            "soft_contact_ke": 1e5,
            "soft_contact_kf": 1e1,
            "soft_contact_kd": 1e2,
            "soft_contact_mu": 0.9,
            "soft_contact_margin": 1e1,
        },
    ]

    results = []

    for i, params in enumerate(sweeps):
        print("Sweep", i)

        env = Bounce(num_envs=N, num_steps=H, std=std, **params)

        # get last model jacobians
        jacs = []
        for i in tqdm(range(env.sim_steps)):
            tape = wp.Tape()
            with tape:
                env.states[i].clear_forces()
                wp.sim.collide(env.model, env.states[i])
                env.integrator.simulate(
                    env.model, env.states[i], env.states[i + 1], env.sim_dt
                )

            # For each timestep compute the jacobian
            # due to the way backprop works, we have to compute it per output dimension
            jacobian = np.empty((N, 6, 6), dtype=np.float32)
            for out_idx in range(3):
                select_index = np.zeros(3)
                select_index[out_idx] = 1.0
                e = wp.array(np.tile(select_index, N), dtype=wp.vec3)
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

        result = {"jacobians": np.array(jacs), "trajectories": env.trajectory()}
        result.update(params)
        results.append(result)

    directory = "outputs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "bounce_grads_{:}".format(H, clip)
    filename = f"{directory}/{filename}"
    print("Saving to", filename)

    np.save("jacobians_sweep", results)


if __name__ == "__main__":
    main()
