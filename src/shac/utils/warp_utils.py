import numpy as np
import torch
import warp as wp
from warp.sim.utils import quat_decompose, quat_twist


PI = wp.constant(np.pi)


@wp.kernel
def assign_kernel(
    b: wp.array(dtype=float),
    # outputs
    a: wp.array(dtype=float),
):
    tid = wp.tid()
    a[tid] = b[tid]


def float_assign(a, b):
    wp.launch(
        assign_kernel,
        dim=len(b),
        device=b.device,
        inputs=[b],
        outputs=[a],
    )
    return a


@wp.kernel
def assign_act_kernel(
    b: wp.array(dtype=float),
    # outputs
    a: wp.array(dtype=float),
):
    tid = wp.tid()
    a[2 * tid] = b[tid]
    a[2 * tid + 1] = 0.0


def float_assign_joint_act(a, b):
    wp.launch(
        assign_act_kernel,
        dim=len(b),
        device=b.device,
        inputs=[b],
        outputs=[a],
    )
    return a


@wp.kernel
def assign_transform_kernel(
    b: wp.array(dtype=wp.transform),
    # outputs
    a: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    a[tid] = b[tid]


def transform_assign(a, b):
    wp.launch(
        assign_transform_kernel,
        dim=len(b),
        device=b.device,
        inputs=[b],
        outputs=[a],
    )
    return a


@wp.kernel
def assign_spatial_kernel(
    b: wp.array(dtype=wp.spatial_vector),
    # outputs
    a: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    a[tid] = b[tid]


def spatial_assign(a, b):
    wp.launch(
        assign_spatial_kernel,
        dim=len(b),
        device=b.device,
        inputs=[b],
        outputs=[a],
    )
    return a


@wp.func
def compute_joint_q(
    X_wp: wp.transform, X_wc: wp.transform, axis: wp.vec3, rotation_count: float
):
    # child transform and moment arm
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)
    # angular error pos
    r_err = wp.quat_inverse(q_p) * q_c

    # swing twist decomposition
    twist = quat_twist(axis, r_err)

    q = (
        wp.acos(twist[3])
        * 2.0
        * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
    ) + 4.0 * PI * rotation_count
    return q


@wp.func
def compute_joint_qd(
    X_wp: wp.transform,
    X_wc: wp.transform,
    w_p: wp.vec3,
    w_c: wp.vec3,
    axis: wp.vec3,
):
    axis_p = wp.transform_vector(X_wp, axis)
    # angular error vel
    w_err = w_c - w_p
    qd = wp.dot(w_err, axis_p)
    return qd


@wp.kernel
def get_joint_q(
    body_q: wp.array(dtype=wp.transform),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_rotation_count: float,
    # outputs
    joint_q: wp.array(dtype=float),
):
    tid = wp.tid()
    type = joint_type[tid]
    axis = joint_axis[tid]

    if type != wp.sim.JOINT_REVOLUTE:
        return

    c_child = tid
    c_parent = joint_parent[tid]
    X_wp = joint_X_p[tid]
    X_wc = body_q[c_child]

    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp

    q = compute_joint_q(X_wp, X_wc, axis, joint_rotation_count)
    joint_q[0] = q


@wp.kernel
def get_joint_qd(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    # outputs
    joint_qd: wp.array(dtype=float),
):
    tid = wp.tid()
    type = joint_type[tid]
    qd_start = joint_qd_start[tid]
    axis = joint_axis[tid]

    c_child = tid
    c_parent = joint_parent[tid]
    X_wp = joint_X_p[tid]
    X_wc = body_q[c_child]
    w_p = wp.vec3()  # is zero if parent is root
    w_c = wp.spatial_top(body_qd[c_child])

    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp
        twist_p = body_qd[c_parent]
        w_p = wp.spatial_top(twist_p)

    qd = compute_joint_qd(X_wp, X_wc, w_p, w_c, axis)

    joint_qd[0] = qd


class IntegratorSimulate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        model,
        state_in,
        integrator,
        dt,
        substeps,
        act,
        body_q,
        body_qd,
        state_out,
    ):
        ctx.tape = wp.Tape()
        ctx.model = model
        ctx.act = wp.from_torch(act)
        ctx.body_q = wp.from_torch(body_q, dtype=wp.transform)
        ctx.body_qd = wp.from_torch(body_qd, dtype=wp.spatial_vector)
        ctx.state_in = state_in

        ctx.joint_q_end = wp.zeros_like(model.joint_q)
        ctx.joint_qd_end = wp.zeros_like(model.joint_qd)

        # record gradients for act, joint_q, and joint_qd
        ctx.act.requires_grad = True
        ctx.joint_q_end.requires_grad = True
        ctx.joint_qd_end.requires_grad = True
        ctx.body_q.requires_grad = True
        ctx.body_qd.requires_grad = True

        with ctx.tape:
            float_assign_joint_act(ctx.model.joint_act, ctx.act)
            # transform_assign(ctx.state_in.body_q, ctx.body_q)
            # spatial_assign(ctx.state_in.body_qd, ctx.body_qd)
            # eval_FK and eval_IK together break body integration due to small errors in
            # revolute joints, therefore DO NOT call in forward pass
            # wp.sim.eval_fk(
            #     ctx.model, ctx.model.joint_q, ctx.model.joint_qd, None, state_in
            # )
            for _ in range(substeps - 1):
                state_in.clear_forces()
                state_temp = model.state(requires_grad=True)
                state_temp = integrator.simulate(
                    ctx.model,
                    state_in,
                    state_temp,
                    dt / float(substeps),
                    requires_grad=True,
                )
                state_in = state_temp
            # updates joint_q joint_qd
            ctx.state_out = integrator.simulate(
                ctx.model, state_in, state_out, dt / float(substeps), requires_grad=True
            )
            # TODO: Check if calling collide after running substeps is correct
            if ctx.model.ground:
                wp.sim.collide(ctx.model, ctx.state_out)
            wp.sim.eval_ik(ctx.model, ctx.state_out, ctx.joint_q_end, ctx.joint_qd_end)

            # wp.launch(
            #     kernel=get_joint_q,
            #     dim=model.joint_count,
            #     device=model.device,
            #     inputs=[
            #         ctx.state_out.body_q,
            #         model.joint_type,
            #         model.joint_parent,
            #         model.joint_X_p,
            #         model.joint_axis,
            #         0.0,
            #     ],
            #     outputs=[ctx.joint_q_end],
            # )
            # wp.launch(
            #     kernel=get_joint_qd,
            #     dim=model.joint_count,
            #     device=model.device,
            #     inputs=[
            #         ctx.state_out.body_q,
            #         ctx.state_out.body_qd,
            #         model.joint_qd_start,
            #         model.joint_type,
            #         model.joint_parent,
            #         model.joint_X_p,
            #         model.joint_axis,
            #     ],
            #     outputs=[ctx.joint_qd_end],
            # )
        joint_q_end = wp.to_torch(ctx.joint_q_end)
        joint_qd_end = wp.to_torch(ctx.joint_qd_end)
        return (
            joint_q_end,
            joint_qd_end,
            ctx.state_out,
        )

    @staticmethod
    def backward(ctx, adj_joint_q, adj_joint_qd, _a):
        # map incoming Torch grads to our output variables
        ctx.joint_q_end.grad = wp.from_torch(adj_joint_q)
        ctx.joint_qd_end.grad = wp.from_torch(adj_joint_qd)

        ctx.tape.backward()
        joint_act_grad = wp.to_torch(ctx.tape.gradients[ctx.act]).clone()
        # Unnecessary copying of grads, grads should already be recorded by context
        body_q_grad = wp.to_torch(ctx.tape.gradients[ctx.state_in.body_q]).clone()
        body_qd_grad = wp.to_torch(ctx.tape.gradients[ctx.state_in.body_qd]).clone()
        # ctx.body_q.grad = wp.from_torch(body_q_grad)
        # ctx.body_qd.grad = wp.from_torch(body_qd_grad)

        ctx.tape.zero()
        # return adjoint w.r.t. inputs
        return (
            None,
            None,
            None,
            None,
            None,
            joint_act_grad,
            body_q_grad,
            body_qd_grad,
            None,
        )


def check_grads(wp_struct):
    for var in wp_struct.__dict__:
        if isinstance(getattr(wp_struct, var), wp.array):
            arr = getattr(wp_struct, var)
            if arr.requires_grad:
                assert np.count_nonzero(arr.grad.numpy()) == 0, "var grad is non_zero"
            else:
                if arr.dtype in [wp.vec3, wp.vec4, float, wp.float32]:
                    print(var)
