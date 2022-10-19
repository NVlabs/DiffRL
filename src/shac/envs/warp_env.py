import numpy as np
import torch
import warp as wp
from gym import spaces


class WarpEnv:
    dt = 1.0 / 60.0
    sim_substeps = 4
    sim_dt = dt
    render_freq = 30  # every half second

    def __init__(
        self,
        num_envs,
        num_obs,
        num_act,
        episode_length,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=False,
        device="cuda",
        env_name="warp_env",
    ):
        self.seed = seed
        self.no_grad = no_grad
        self.device = str(device).split(":")[0]
        self.visualize = render
        self.stochastic_init = stochastic_init
        self.sim_time = 0.0
        self.num_frames = 0

        self.num_environments = num_envs
        self.env_name = env_name

        # initialize observation and action space
        self.num_observations = num_obs
        self.num_actions = num_act
        self.episode_length = episode_length

        self.obs_space = spaces.Box(
            np.ones(self.num_observations) * -np.Inf,
            np.ones(self.num_observations) * np.Inf,
        )
        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0
        )

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations),
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float, requires_grad=False
        )
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        # end of the episode
        self.termination_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions),
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )

        self.eval_fk_mask = None

        self.extras = {}

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

    def get_state(self, return_act=False, reshape=False):
        """Returns model joint state (position and velocity)"""
        joint_q, joint_qd = (
            wp.to_torch(self.model.joint_q),
            wp.to_torch(self.model.joint_qd),
        )
        if reshape:
            joint_q, joint_qd = joint_q.view(self.num_envs, -1), joint_qd.view(self.num_envs, -1)
        if not return_act:
            return joint_q, joint_qd
        joint_act = wp.to_torch(self.model.joint_act)
        return joint_q, joint_qd, joint_act

    def calculateObservations(self):
        """
        Calculate the observations for the current state
        """
        raise NotImplementedError

    def calculateReward(self):
        """
        Calculate the reward for the current state
        """
        raise NotImplementedError

    def get_stochastic_init(self, env_ids, joint_q, joint_qd):
        """
        Get the rand initial state for the environment
        """
        raise NotImplementedError

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(
                    self.num_envs, dtype=torch.long, device=self.device
                )
        if len(env_ids) == self.num_envs:
            # clear_grad needs to be called before to zero grads
            self.clear_grad()
        if env_ids is not None:
            # fixed start state
            self.joint_q, self.joint_qd, joint_act = self.get_state(return_act=True)
            joint_act = torch.zeros_like(self.joint_q, device=self.device)
            joint_q, joint_qd = self.joint_q.view(self.num_envs, -1), self.joint_qd.view(self.num_envs, -1)
            joint_act = joint_act.view(self.num_envs, -1)
            assert len(self.start_joint_q) == self.num_joint_q, f"got shape {self.start_joint_q.shape}, not {self.num_joint_q}"
            joint_q[env_ids] = self.start_joint_q.clone()
            joint_qd[env_ids] = self.start_joint_qd.clone()
            joint_act[env_ids] = self.start_joint_act.clone()

            if self.stochastic_init:
                joint_q, joint_qd = self.get_stochastic_init(env_ids, joint_q, joint_qd)

            requires_grad = not self.no_grad
            # checks if requires_grad set properly in clear_grad
            self.joint_q.requires_grad = requires_grad
            self.joint_qd.requires_grad = requires_grad
            joint_act.requires_grad = requires_grad
            joint_act = joint_act.view(-1)
            self.model.joint_q.assign(wp.from_torch(self.joint_q))
            self.model.joint_qd.assign(wp.from_torch(self.joint_qd))
            self.model.joint_act.assign(wp.from_torch(joint_act))
            self.state = self.model.state(requires_grad=requires_grad)
            # updates state body positions after reset
            # TODO: does this also pass through gradients between state.body_q/qd to
            #       joint_q?
            wp.sim.eval_fk(
                self.model, self.model.joint_q, self.model.joint_qd, self.eval_fk_mask, self.state
            )
            if self.model.ground:
                # Called once at the beginning of trajectory to initialize contact body/body mesh/esh pairs
                self.model.collide(self.state)
            # reset progress buffer (i.e. episode done flag)
            self.progress_buf[env_ids] = 0
            self.calculateObservations()

        return self.obs_buf

    def clear_grad(self):
        """
        cut off the gradient from the current state to previous states
        """
        with torch.no_grad():  # TODO: check with Miles
            current_joint_q = wp.to_torch(self.model.joint_q).detach()
            current_joint_qd = wp.to_torch(self.model.joint_qd).detach()
            current_joint_act = wp.to_torch(self.model.joint_act).detach()
            requires_grad = not self.no_grad
            self.model.joint_q.assign(wp.from_torch(current_joint_q))
            self.model.joint_qd.assign(wp.from_torch(current_joint_qd))
            self.model.joint_act.assign(wp.from_torch(current_joint_act))
            self.joint_q = None
            self.state = self.model.state(requires_grad=(not self.no_grad))
        if not self.no_grad:
            self.model.joint_q.requires_grad = True
            self.model.joint_qd.requires_grad = True
            self.model.joint_act.requires_grad = True
            self.model.body_q.requires_grad = True
            self.model.body_qd.requires_grad = True
            self.state.body_q.requires_grad = True
            self.state.body_qd.requires_grad = True

    def step(self, act):
        """
        Step the simulation forward by one timestep
        """
        raise NotImplementedError