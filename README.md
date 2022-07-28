# SHAC

This repository contains the implementation for the paper [Accelerated Policy Learning with Parallel Differentiable Simulation](https://short-horizon-actor-critic.github.io/) (ICLR 2022).



In this paper, we present a GPU-based differentiable simulation and propose a policy learning method named SHAC leveraging the developed differentiable simulation. We provide a comprehensive benchmark set for policy learning with differentiable simulation. The benchmark set contains six robotic control problems for now as shown in the figure below. 

<p align="center">
    <img src="figures/envs.png" alt="envs" width="800" />
</p>

## Installation

- `git clone https://github.com/NVlabs/DiffRL.git --recursive`

- The code has been tested on 
  - Operating System: Ubuntu 16.04, 18.04, 20.04, 21.10, 22.04
  - Python Version: 3.7, 3.8
  - GPU: TITAN X, RTX 1080, RTX 2080, RTX 3080, RTX 3090, RTX 3090 Ti

#### Prerequisites

- In the project folder, create a virtual environment in Anaconda:

  ```
  conda env create -f diffrl_conda.yml
  conda activate shac
  ```

- dflex

  ```
  cd dflex
  pip install -e .
  ```

- rl_games, forked from [rl-games](https://github.com/Denys88/rl_games) (used for PPO and SAC training):

  ````
  cd externals/rl_games
  pip install -e .
  ````

- Install an older version of protobuf required for TensorboardX:
  ````
  pip install protobuf==3.20.0
  ````

#### Test Examples

A test example can be found in the `examples` folder.

```
python test_env.py --env AntEnv
```

If the console outputs `Finish Successfully` in the last line, the code installation succeeds.


## Training

Running the following commands in `examples` folder allows to train Ant with SHAC.
```
python train_shac.py --cfg ./cfg/shac/ant.yaml --logdir ./logs/Ant/shac
```

We also provide a one-line script in the `examples/train_script.sh` folder to replicate the results reported in the paper for both our method and for baseline method. The results might slightly differ from the paper due to the randomness of the cuda and different Operating System/GPU/Python versions. The plot reported in paper is produced with TITAN X on Ubuntu 16.04.

#### SHAC (Our Method)

For example, running the following commands in `examples` folder allows to train Ant and SNU Humanoid (Humanoid MTU in the paper) environments with SHAC respectively for 5 individual seeds.

```
python train_script.py --env Ant --algo shac --num-seeds 5
```

```
python train_script.py --env SNUHumanoid --algo shac --num-seeds 5
```

#### Baseline Algorithms

For example, running the following commands in `examples` folder allows to train Ant environment with PPO implemented in RL_games for 5 individual seeds,

```
python train_script.py --env Ant --algo ppo --num-seeds 5
```

## Testing

To test the trained policy, you can input the policy checkpoint into the training script and use a `--play` flag to indicate it is for testing. For example, the following command allows to test a trained policy (assume the policy is located in `logs/Ant/shac/policy.pt`)

```
python train_shac.py --cfg ./cfg/shac/ant.yaml --checkpoint ./logs/Ant/shac/policy.pt --play [--render]
```

The `--render` flag indicates whether to export the video of the task execution. If does, the exported video is encoded in `.usd` format, and stored in the `examples/output` folder. To visualize the exported `.usd` file, refer to [USD at NVIDIA](https://developer.nvidia.com/usd).

## Citation

If you find our paper or code is useful, please consider citing:
```kvk
  @inproceedings{xu2021accelerated,
    title={Accelerated Policy Learning with Parallel Differentiable Simulation},
    author={Xu, Jie and Makoviychuk, Viktor and Narang, Yashraj and Ramos, Fabio and Matusik, Wojciech and Garg, Animesh and Macklin, Miles},
    booktitle={International Conference on Learning Representations},
    year={2021}
  }
```