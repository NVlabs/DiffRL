class RigidBodySimulator:
    """
    Differentiable simulator of a rigid-body system with contacts.
    The system state is described entirely by the joint positions q, velocities qd, and
    joint torques tau. The system state is updated by calling the warp_step() function.
    """

    frame_dt = 1.0/60.0

    episode_frames = 240

    sim_substeps = 1
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    name = 'single_cartpole'
    urdf_path = "assets/cartpole_single.urdf"
    q_init = [0., 0.3],

    def __init__(self, render=False, num_envs=1, device='cpu'): 
        self.device = device
        self.render = render

        self.num_envs = num_envs
        self.init_sim()

    def init_sim(self):
        self.builder = wp.sim.ModelBuilder()
        for i in range(self.num_envs):
            wp.sim.parse_urdf(os.path.join(os.path.dirname(__file__), self.urdf), self.builder,
                xform=wp.transform(np.array((0.0, 0.0, 0.0)),
                    wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
                floating=False, 
                density=0,
                armature=0.1,
                stiffness=0.0,
                damping=0.0,
                shape_ke=1.e+4,
                shape_kd=1.e+2,
                shape_kf=1.e+2,
                shape_mu=1.0,
                limit_ke=1.e+4,
                limit_kd=1.e+1)
            # joint initial positions
            builder.joint_q[len(self.q_init):] = self.q_init
            builder.joint_target[:len(self.q_init)] = [0.0 for _ in range(len(self.q_init))]

        # finalize model
        self.model = builder.finalize(device)
        self.builder = builder
        self.model.ground = False

        if self.use_single_cartpole:
            self.model.joint_attach_ke = 40000.0
            self.model.joint_attach_kd = 200.0
        else:
            self.model.joint_attach_ke = 1600.0
            self.model.joint_attach_kd = 20.0


        self.dof_q = self.model.joint_coord_count
        self.dof_qd = self.model.joint_dof_count

        self.state = self.model.state()

        self.solve_iterations = 10
        # self.integrator = wp.sim.XPBDIntegrator(self.solve_iterations)
        self.integrator = wp.sim.SemiImplicitIntegrator()
        
        if (self.model.ground):
            self.model.collide(self.state)

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__), f"outputs/{self.name}.usd"))

    def warp_step(self, q, qd, tau, q_next, qd_next, requires_grad=False):
        if requires_grad:
            # ground = self.model.ground
            # self.model = self.builder.finalize(self.device)
            # self.model.ground = ground

            self.model.joint_act.requires_grad = True
            self.model.body_q.requires_grad = True
            self.model.body_qd.requires_grad = True

            self.model.body_mass.requires_grad = True
            self.model.body_inv_mass.requires_grad = True
            self.model.body_inertia.requires_grad = True
            self.model.body_inv_inertia.requires_grad = True
            self.model.body_com.requires_grad = True

            # just enable requires_grad for all arrays in the model
            # for name in dir(self.model):
            #     attr = getattr(self.model, name)
            #     if isinstance(attr, wp.array):
            #         attr.requires_grad = True
            
            # XXX activate requires_grad for all arrays in the material struct
            self.model.shape_materials.ke.requires_grad = True
            self.model.shape_materials.kd.requires_grad = True
            self.model.shape_materials.kf.requires_grad = True
            self.model.shape_materials.mu.requires_grad = True
            self.model.shape_materials.restitution.requires_grad = True
            
            states = [self.model.state(requires_grad=True) for _ in range(self.sim_substeps+1)]
        else:
            # states = [self.state for _ in range(self.sim_substeps+1)]
            states = [self.model.state(requires_grad=False) for _ in range(self.sim_substeps+1)]

        wp.sim.eval_fk(self.model, q, qd, None, states[0])

        # assign input controls as joint torques
        wp.launch(inplace_assign, dim=self.dof_qd, inputs=[tau], outputs=[self.model.joint_act], device=self.device)
        
        for i in range(self.sim_substeps):
            states[i].clear_forces()
            if self.model.ground:
                self.model.allocate_rigid_contacts()
                wp.sim.collide(self.model, states[i])
            self.integrator.simulate(self.model, states[i], states[i+1], self.sim_dt)

        wp.sim.eval_ik(self.model, states[-1], q_next, qd_next)    


