import dflex
import dflex.adjoint as df
import torch

from dflex.model import *
from dflex.sim import SemiImplicitIntegrator
from dflex.sim import (
    eval_springs,
    eval_triangles,
    eval_triangles_contact,
    eval_bending,
    eval_contacts,
    eval_tetrahedra,
    eval_rigid_fk,
    eval_rigid_id,
    eval_soft_contacts,
    eval_muscles,
    eval_rigid_tau,
    eval_rigid_jacobian,
    eval_rigid_mass,
    eval_dense_cholesky_batched,
    eval_dense_solve_batched,
    eval_rigid_integrate,
    integrate_particles,
)


def eval_rigid_contacts_art(
    body_X_s: df.tensor(df.spatial_transform),
    body_v_s: df.tensor(df.spatial_vector),
    contact_body: df.tensor(int),
    contact_point: df.tensor(df.float3),
    contact_dist: df.tensor(float),
    contact_mat: df.tensor(int),
    materials: df.tensor(float),
    body_f_s: df.tensor(df.spatial_vector),
    active_contacts: df.tensor(bool),
):
    tid = df.tid()

    c_body = df.load(contact_body, tid)
    c_point = df.load(contact_point, tid)
    c_dist = df.load(contact_dist, tid)
    c_mat = df.load(contact_mat, tid)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = df.load(materials, c_mat * 4 + 0)  # restitution coefficient
    kd = df.load(materials, c_mat * 4 + 1)  # damping coefficient
    kf = df.load(materials, c_mat * 4 + 2)  # friction coefficient
    mu = df.load(materials, c_mat * 4 + 3)  # coulomb friction

    X_s = df.load(body_X_s, c_body)  # position of colliding body
    v_s = df.load(body_v_s, c_body)  # orientation of colliding body

    n = float3(0.0, 1.0, 0.0)

    # transform point to world space
    p = (
        df.spatial_transform_point(X_s, c_point) - n * c_dist
    )  # add on 'thickness' of shape, e.g.: radius of sphere/capsule

    w = df.spatial_top(v_s)
    v = df.spatial_bottom(v_s)

    # contact point velocity
    dpdt = v + df.cross(w, p)

    # check ground contact
    c = df.dot(n, p)  # check if we're inside the ground

    if c >= 0.0:
        active_contacts[c_body] = False
        return

    vn = dot(n, dpdt)  # velocity component out of the ground
    vt = dpdt - n * vn  # velocity component not into the ground

    fn = c * ke  # normal force (restitution coefficient * how far inside for ground)

    # contact damping
    fd = df.min(vn, 0.0) * kd * df.step(c) * (0.0 - c)

    # viscous friction
    # ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)  # negative
    upper = 0.0 - lower  # positive, workaround for no unary ops

    vx = df.clamp(dot(float3(kf, 0.0, 0.0), vt), lower, upper)
    vz = df.clamp(dot(float3(0.0, 0.0, kf), vt), lower, upper)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    ft = df.normalize(vt) * df.min(kf * df.length(vt), 0.0 - mu * c * ke) * df.step(c)

    f_total = n * (fn + fd) + ft
    t_total = df.cross(p, f_total)

    active_contacts[c_body] = True
    df.atomic_add(body_f_s, c_body, df.spatial_vector(t_total, f_total))


class ContactSemiImplicitIntegrator(SemiImplicitIntegrator):
    def state(self) -> State:
        state = super().state()
        state.active_contacts = torch.zeros(state.body_f_s.shape[0], dtype=torch.bool)
        return state

    def _simulate(self, tape, model, state_in, state_out, dt, update_mass_matrix=True):
        with dflex.util.ScopedTimer("simulate", False):
            # alloc particle force buffer
            if model.particle_count:
                state_out.particle_f.zero_()

            if model.link_count:
                state_out.body_ft_s = torch.zeros(
                    (model.link_count, 6),
                    dtype=torch.float32,
                    device=model.adapter,
                    requires_grad=True,
                )
                state_out.body_f_ext_s = torch.zeros(
                    (model.link_count, 6),
                    dtype=torch.float32,
                    device=model.adapter,
                    requires_grad=True,
                )

            # damped springs
            if model.spring_count:
                tape.launch(
                    func=eval_springs,
                    dim=model.spring_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.spring_indices,
                        model.spring_rest_length,
                        model.spring_stiffness,
                        model.spring_damping,
                    ],
                    outputs=[state_out.particle_f],
                    adapter=model.adapter,
                )

            # triangle elastic and lift/drag forces
            if model.tri_count and model.tri_ke > 0.0:
                tape.launch(
                    func=eval_triangles,
                    dim=model.tri_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_activations,
                        model.tri_ke,
                        model.tri_ka,
                        model.tri_kd,
                        model.tri_drag,
                        model.tri_lift,
                    ],
                    outputs=[state_out.particle_f],
                    adapter=model.adapter,
                )

            # triangle/triangle contacts
            if model.enable_tri_collisions and model.tri_count and model.tri_ke > 0.0:
                tape.launch(
                    func=eval_triangles_contact,
                    dim=model.tri_count * model.particle_count,
                    inputs=[
                        model.particle_count,
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_activations,
                        model.tri_ke,
                        model.tri_ka,
                        model.tri_kd,
                        model.tri_drag,
                        model.tri_lift,
                    ],
                    outputs=[state_out.particle_f],
                    adapter=model.adapter,
                )

            # triangle bending
            if model.edge_count:
                tape.launch(
                    func=eval_bending,
                    dim=model.edge_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.edge_indices,
                        model.edge_rest_angle,
                        model.edge_ke,
                        model.edge_kd,
                    ],
                    outputs=[state_out.particle_f],
                    adapter=model.adapter,
                )

            # particle ground contact
            if model.ground and model.particle_count:
                tape.launch(
                    func=eval_contacts,
                    dim=model.particle_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.contact_ke,
                        model.contact_kd,
                        model.contact_kf,
                        model.contact_mu,
                    ],
                    outputs=[state_out.particle_f],
                    adapter=model.adapter,
                )

            # tetrahedral FEM
            if model.tet_count:
                tape.launch(
                    func=eval_tetrahedra,
                    dim=model.tet_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        model.tet_indices,
                        model.tet_poses,
                        model.tet_activations,
                        model.tet_materials,
                    ],
                    outputs=[state_out.particle_f],
                    adapter=model.adapter,
                )

            # ----------------------------
            # articulations

            if model.link_count:
                # evaluate body transforms
                tape.launch(
                    func=eval_rigid_fk,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_joint_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        model.joint_X_pj,
                        model.joint_X_cm,
                        model.joint_axis,
                    ],
                    outputs=[state_out.body_X_sc, state_out.body_X_sm],
                    adapter=model.adapter,
                    preserve_output=True,
                )

                # evaluate joint inertias, motion vectors, and forces
                tape.launch(
                    func=eval_rigid_id,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_joint_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        state_in.joint_qd,
                        model.joint_axis,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.body_I_m,
                        state_out.body_X_sc,
                        state_out.body_X_sm,
                        model.joint_X_pj,
                        model.gravity,
                    ],
                    outputs=[
                        state_out.joint_S_s,
                        state_out.body_I_s,
                        state_out.body_v_s,
                        state_out.body_f_s,
                        state_out.body_a_s,
                    ],
                    adapter=model.adapter,
                    preserve_output=True,
                )

                if model.ground and model.contact_count > 0:
                    # evaluate contact forces
                    tape.launch(
                        func=eval_rigid_contacts_art,
                        dim=model.contact_count,
                        inputs=[
                            state_out.body_X_sc,
                            state_out.body_v_s,
                            model.contact_body0,
                            model.contact_point0,
                            model.contact_dist,
                            model.contact_material,
                            model.shape_materials,
                        ],
                        outputs=[state_out.body_f_s, state_out.active_contacts],
                        adapter=model.adapter,
                        preserve_output=True,
                    )

                # particle shape contact
                if model.particle_count:
                    # tape.launch(func=eval_soft_contacts,
                    #             dim=model.particle_count*model.shape_count,
                    #             inputs=[state_in.particle_q, state_in.particle_qd, model.contact_ke, model.contact_kd, model.contact_kf, model.contact_mu],
                    #             outputs=[state_out.particle_f],
                    #             adapter=model.adapter)

                    tape.launch(
                        func=eval_soft_contacts,
                        dim=model.particle_count * model.shape_count,
                        inputs=[
                            model.particle_count,
                            state_in.particle_q,
                            state_in.particle_qd,
                            state_in.body_X_sc,
                            state_in.body_v_s,
                            model.shape_transform,
                            model.shape_body,
                            model.shape_geo_type,
                            torch.Tensor(),
                            model.shape_geo_scale,
                            model.shape_materials,
                            model.contact_ke,
                            model.contact_kd,
                            model.contact_kf,
                            model.contact_mu,
                        ],
                        # outputs
                        outputs=[state_out.particle_f, state_out.body_f_s],
                        adapter=model.adapter,
                    )

                # evaluate muscle actuation
                tape.launch(
                    func=eval_muscles,
                    dim=model.muscle_count,
                    inputs=[
                        state_out.body_X_sc,
                        state_out.body_v_s,
                        model.muscle_start,
                        model.muscle_params,
                        model.muscle_links,
                        model.muscle_points,
                        model.muscle_activation,
                    ],
                    outputs=[state_out.body_f_s],
                    adapter=model.adapter,
                    preserve_output=True,
                )

                # evaluate joint torques
                tape.launch(
                    func=eval_rigid_tau,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_joint_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        state_in.joint_qd,
                        state_in.joint_act,
                        model.joint_target,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_limit_ke,
                        model.joint_limit_kd,
                        model.joint_axis,
                        state_out.joint_S_s,
                        state_out.body_f_s,
                    ],
                    outputs=[state_out.body_ft_s, state_out.joint_tau],
                    adapter=model.adapter,
                    preserve_output=True,
                )

                if update_mass_matrix:
                    model.alloc_mass_matrix()

                    # build J
                    tape.launch(
                        func=eval_rigid_jacobian,
                        dim=model.articulation_count,
                        inputs=[
                            # inputs
                            model.articulation_joint_start,
                            model.articulation_J_start,
                            model.joint_parent,
                            model.joint_qd_start,
                            state_out.joint_S_s,
                        ],
                        outputs=[model.J],
                        adapter=model.adapter,
                        preserve_output=True,
                    )

                    # build M
                    tape.launch(
                        func=eval_rigid_mass,
                        dim=model.articulation_count,
                        inputs=[
                            # inputs
                            model.articulation_joint_start,
                            model.articulation_M_start,
                            state_out.body_I_s,
                        ],
                        outputs=[model.M],
                        adapter=model.adapter,
                        preserve_output=True,
                    )

                    # form P = M*J
                    df.matmul_batched(
                        tape,
                        model.articulation_count,
                        model.articulation_M_rows,
                        model.articulation_J_cols,
                        model.articulation_J_rows,
                        0,
                        0,
                        model.articulation_M_start,
                        model.articulation_J_start,
                        model.articulation_J_start,  # P start is the same as J start since it has the same dims as J
                        model.M,
                        model.J,
                        model.P,
                        adapter=model.adapter,
                    )

                    # form H = J^T*P
                    df.matmul_batched(
                        tape,
                        model.articulation_count,
                        model.articulation_J_cols,
                        model.articulation_J_cols,
                        model.articulation_J_rows,  # P rows is the same as J rows
                        1,
                        0,
                        model.articulation_J_start,
                        model.articulation_J_start,  # P start is the same as J start since it has the same dims as J
                        model.articulation_H_start,
                        model.J,
                        model.P,
                        model.H,
                        adapter=model.adapter,
                    )

                    # compute decomposition
                    tape.launch(
                        func=eval_dense_cholesky_batched,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_H_start,
                            model.articulation_H_rows,
                            model.H,
                            model.joint_armature,
                        ],
                        outputs=[model.L],
                        adapter=model.adapter,
                        skip_check_grad=True,
                    )

                tmp = torch.zeros_like(state_out.joint_tau)

                # solve for qdd
                tape.launch(
                    func=eval_dense_solve_batched,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_dof_start,
                        model.articulation_H_start,
                        model.articulation_H_rows,
                        model.H,
                        model.L,
                        state_out.joint_tau,
                        tmp,
                    ],
                    outputs=[state_out.joint_qdd],
                    adapter=model.adapter,
                    skip_check_grad=True,
                )

                # integrate joint dofs -> joint coords
                tape.launch(
                    func=eval_rigid_integrate,
                    dim=model.link_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        state_in.joint_qd,
                        state_out.joint_qdd,
                        dt,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd],
                    adapter=model.adapter,
                )

            # ----------------------------
            # integrate particles

            if model.particle_count:
                tape.launch(
                    func=integrate_particles,
                    dim=model.particle_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        state_out.particle_f,
                        model.particle_inv_mass,
                        model.gravity,
                        dt,
                    ],
                    outputs=[state_out.particle_q, state_out.particle_qd],
                    adapter=model.adapter,
                )

            return state_out
