from enum import Enum

import warp as wp

from .warp_env import WarpEnv


class GoalType(Enum):
    POSITION = 0
    ORIENTATION = 1


class ClawWarpEnv(WarpEnv):
    body_names = [
        "root",
        "right_finger",
        "proximal_body_right_finger",
        "distal_body_right_finger",
        "left_finger",
        "proximal_body_left_finger",
        "distal_body_left_finger",
        "object",
        "goal",
    ]

    def __init__(
        self,
        num_envs=1,
        episode_length=100,
        seed=42,
        render=False,
        device="cuda",
        stochastic_init=False,
        goal_type=GoalType.POSITION,
    ):
        num_obs = 22
        num_act = 4
        self.act_joints_indices = np.array([2, 3, 5, 6])  # indexes finger joints
        super(ClawWarpEnv, self).__init__(
            num_envs, num_obs, num_act, episode_length, seed, render, device
        )
        self.init_sim()

        # -----------------------
        # set up Usd renderer
        if self.visualize:
            self.renderer = wp.sim.render.SimRenderer(
                self.model,
                osp.join(osp.dirname(__file__), f"outputs/Claw_{self.num_envs}.usd"),
            )
        self.stochastic_init = self.stochastic_init
        self.goal_type = goal_type

    def init_sim(self):
        self.num_joint_q, self.num_joint_qd = 7, 7
        self.articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_mjcf(
            "/home/ksrini/shac/envs/assets/claw.xml",
            self.articulation_builder,
            density=0,
            armature_scale=10.0,
            stiffness=0.0,
            damping=0.1,
            contact_ke=1.0e4,
            contact_kd=1.0e2,
            contact_kf=1.0e2,
            contact_mu=0.5,
            limit_ke=1.0e2,
            limit_kd=1.0e1,
        )
        self.builder = wp.sim.ModelBuilder()
        for i in range(self.num_envs):
            # Puts space between parallel envs
            self.builder.add_rigid_articulation(
                self.articulation_builder,
                xform=wp.transform(
                    np.array([i * 2, 0.0, 0.0]),
                    wp.quat_from_axis_angle((1.0, 0.0, 0.0), -np.pi * 0.5),
                ),
            )
            # goal = (0.45, -0.05, -0.323)
            goal = np.zeros(2)
            while True:
                cylinder_pos = np.array(
                    [
                        np.random.uniform(low=-0.3, high=0, size=1),
                        np.random.uniform(low=-0.2, high=0.2, size=1),
                    ]
                )
                if np.linalg.norm(cylinder_pos - goal) > 0.17:
                    break
            # TODO: add task-specific randomization to
            self.builder.joint_q[-6:-4] = cylinder_pos
            self.builder.joint_q[-3:-1] = goal
            self.model = self.builder.finalize(self.device)
            self.model.ground = True
            self.model.joint_attach_ke *= 8.0
            self.model.joint_attach_ke *= 2.0
            self.integrator = wp.sim.SemiImplicitIntegrator()
            self.state = self.model.state()

    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()

        return self.obs_buf

    def get_checkpoint(self):
        checkpoint = {}
        joint_q, joint_qd = self.get_state()
        checkpoint["joint_q"] = joint_q
        checkpoint["joint_qd"] = joint_qd
        checkpoint["actions"] = self.actions.clone()
        checkpoint["progress_buf"] = self.progress_buf.clone()

        return checkpoint

    def calculateObservations(self):
        joint_q, joint_qd = self.get_state()
        x = joint_q[:, 0:2]
        theta = joint_q[:, 2:3]
        xdot = joint_qd[:, 0:2]
        theta_dot = joint_qd[:, 2:3]

        # observations: [x, xdot, theta, theta_dot]
        self.obs_buf = torch.cat([x, xdot, theta, theta_dot], dim=-1)

    def calculateReward(self):
        if self.goal_type is GoalType.POSITION:
            reward_dist = -torch.linalg.norm(self.goal - self._get_obj_pos())
        elif self.goal_type is GoalType.ORIENTATION:
            reward_dist = -np.abs(self.goal - self._get_obj_ori())
        reward_near = -torch.linalg.norm(
            self._get_fingertip_pos() - self._get_obj_pos()
        )
        reward_ctrl = -torch.linalg.norm(action)
        self.rew_buf = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        self.extras.update(
            {
                "reward_dist": reward_dist,
                "reward_near": reward_near,
                "reward_ctrl": reward_ctrl,
                "reward_total": reward_total,
            }
        )

    def step(self, actions):
        assert (
            np.prod(actions.shape) == self.num_envs * self.num_act
        ), f"actions should have compatible shape ({self.num_envs}, self.{num_act}), got {actions.shape}"

        with wp.ScopedTimer("simulate", active=False, detailed=False):

            actions = actions.view((self.num_envs, self.num_act))
            actions = torch.clip(actions, -1.0, 1.0)
            self.actions = actions
            joint_act_t = torch.zeros(
                (self.num_envs, self.num_actions), dtype=torch.float, device=self.device
            )
            joint_act_t[:, self.act_joints_indices] = actions
            self.model.joint_act.assign(joint_act_t)
            # TODO: other methods to assign/copy values from tensor to warp array.
            #       need to check which preserve gradients from input
            # wp.copy(self.model.joint_act, wp.from_torch(joint_act))
            # self.model.joint_act.view(self.num_envs, -1)[:, 0:1] = actions
            self.state = self.integrator.simulate(
                self.model, self.state, self.state, self.sim_dt
            )
            self.sim_time += self.sim_dt
        self.reset_buf = torch.zeros_like(self.reset_buf)
        self.extras = {}
        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateRewards()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
