import warp as wp
import numpy as np
import torch

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
        joint_q_start,
        joint_qd_start,
    ):
        ctx.tape = wp.Tape()
        ctx.model = model
        ctx.act = wp.from_torch(act)
        ctx.joint_q_start = wp.from_torch(joint_q_start)
        ctx.joint_qd_start = wp.from_torch(joint_qd_start)

        ctx.joint_q_end = wp.zeros_like(model.joint_q)
        ctx.joint_qd_end = wp.zeros_like(model.joint_qd)

        # record gradients for act, joint_q, and joint_qd
        ctx.act.requires_grad = True
        ctx.joint_q_start.requires_grad = True
        ctx.joint_qd_start.requires_grad = True
        ctx.joint_q_end.requires_grad = True
        ctx.joint_qd_end.requires_grad = True

        ctx.model.shape_materials.ke.requires_grad = True
        ctx.model.shape_materials.kd.requires_grad = True
        ctx.model.shape_materials.kf.requires_grad = True
        ctx.model.shape_materials.mu.requires_grad = True
        ctx.model.shape_materials.restitution.requires_grad = True 

        with ctx.tape:
            # float_assign(ctx.model.joint_q, ctx.joint_q_start)
            # float_assign(ctx.model.joint_qd, ctx.joint_qd_start)

            wp.sim.eval_fk(
                ctx.model, ctx.joint_q_start, ctx.joint_qd_start, None, state_in
            )
            for _ in range(substeps):
                float_assign_joint_act(ctx.model.joint_act, ctx.act)
                state_in.clear_forces()
                state_temp = model.state(requires_grad=True)
                state_temp = integrator.simulate(
                    model, state_in, state_temp, dt / float(substeps)
                )
                state_in = state_temp

            # updates body position/vel
            ctx.state_out = state_temp
            # updates joint_q joint_qd
            wp.sim.eval_ik(ctx.model, ctx.state_out, ctx.joint_q_end, ctx.joint_qd_end)
        return (
            wp.to_torch(ctx.joint_q_end),
            wp.to_torch(ctx.joint_qd_end),
            ctx.state_out,
        )

    @staticmethod
    def backward(ctx, adj_joint_q, adj_joint_qd, _):

        # map incoming Torch grads to our output variables
        ctx.joint_q_end.grad = wp.from_torch(adj_joint_q)
        ctx.joint_qd_end.grad = wp.from_torch(adj_joint_qd)

        ctx.tape.backward()
        joint_act_grad = wp.to_torch(ctx.tape.gradients[ctx.act]).clone()
        joint_q_grad = wp.to_torch(ctx.tape.gradients[ctx.joint_q_start]).clone()
        joint_qd_grad = wp.to_torch(ctx.tape.gradients[ctx.joint_qd_start]).clone()
        print(f"joint_act_grad, {joint_act_grad}")
        print(f"joint_q_grad, {joint_q_grad}")
        print(f"joint_qd_grad, {joint_qd_grad}")
        
        ctx.tape.zero()
        # return adjoint w.r.t. inputs
        return (
            None,
            None,
            None,
            None,
            None,
            joint_act_grad,
            joint_q_grad,
            joint_qd_grad,
        )


def check_grads(wp_struct):
    for var in wp_struct.__dict__:
        if isinstance(getattr(wp_struct, var), wp.array):
            arr = getattr(wp_struct, var)
            if arr.requires_grad:
                assert (np.count_nonzero(arr.grad.numpy()) == 0), "var grad is non_zero"
            else:
                if arr.dtype in [wp.vec3, wp.vec4, float, wp.float32]:
                    print(var)

