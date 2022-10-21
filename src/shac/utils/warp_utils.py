import numpy as np
import torch
import warp as wp


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
    b: wp.array2d(dtype=float),
    # outputs
    a: wp.array(dtype=float),
):
    tid = wp.tid()
    a[2 * tid] = b[tid, 0]
    a[2 * tid + 1] = 0.


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

class IntegratorSimulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                model,
                state_in,
                integrator,
                dt,
                substeps,
                act,
                body_q,
                body_qd,
                state_out,):
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

        ctx.model.shape_materials.ke.requires_grad = True
        ctx.model.shape_materials.kd.requires_grad = True
        ctx.model.shape_materials.kf.requires_grad = True
        ctx.model.shape_materials.mu.requires_grad = True
        ctx.model.shape_materials.restitution.requires_grad = True 

        with ctx.tape:
            float_assign_joint_act(ctx.model.joint_act, ctx.act)
            # transform_assign(ctx.state_in.body_q, ctx.body_q)
            # spatial_assign(ctx.state_in.body_qd, ctx.body_qd)
            # eval_FK and eval_IK together break body integration due to small errors in
            # revolute joints, therefore DO NOT call in forward pass
            # wp.sim.eval_fk(ctx.model, ctx.model.joint_q, ctx.model.joint_qd, None, state_in)
            for _ in range(substeps - 1):
                state_in.clear_forces()
                state_temp = model.state(requires_grad=True)
                state_temp = integrator.simulate(
                    ctx.model, state_in, state_temp, dt / float(substeps),
                    requires_grad=True
                )
                state_in = state_temp
            state_in.clear_forces()
            # updates joint_q joint_qd
            ctx.state_out = integrator.simulate(ctx.model, state_in, state_out, dt / float(substeps), requires_grad=True)
            # TODO: Check if calling collide after running substeps is correct
            if ctx.model.ground:
                wp.sim.collide(ctx.model, ctx.state_out)
            wp.sim.eval_ik(ctx.model, ctx.state_out, ctx.joint_q_end, ctx.joint_qd_end)


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
        ctx.body_q.grad = wp.from_torch(body_q_grad)
        ctx.body_qd.grad = wp.from_torch(body_qd_grad)

        ctx.tape.zero()
        # return adjoint w.r.t. inputs
        return (None, None, None, None, None, joint_act_grad, body_q_grad, body_qd_grad, None)


def check_grads(wp_struct):
    for var in wp_struct.__dict__:
        if isinstance(getattr(wp_struct, var), wp.array):
            arr = getattr(wp_struct, var)
            if arr.requires_grad:
                assert (np.count_nonzero(arr.grad.numpy()) == 0), "var grad is non_zero"
            else:
                if arr.dtype in [wp.vec3, wp.vec4, float, wp.float32]:
                    print(var)

