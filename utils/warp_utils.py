import warp as wp
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
def assign_kernel_2d(
    b: wp.array2d(dtype=float),
    # outputs
    a: wp.array(dtype=float),
):
    tid = wp.tid()
    a[2 * tid] = b[tid, 0]
    a[2 * tid + 1] = b[tid, 1]


def float_assign_2d(a, b):
    wp.launch(
        assign_kernel_2d,
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


def float_assign_joint_act(a, b):
    wp.launch(
        assign_act_kernel,
        dim=len(b),
        device=b.device,
        inputs=[b],
        outputs=[a],
    )
    return a


class KernelAutogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        integrator,
        model,
        state_in,
        dt,
        substeps,
        *tensors,
    ):
        ctx.tape = wp.Tape()
        ctx.model = model
        ctx.inputs = [wp.from_torch(t) for t in tensors]
        # allocate output

        with ctx.tape:
            float_assign_joint_act(ctx.model.joint_act, ctx.act)
            float_assign_2d(ctx.model.joint_q, ctx.joint_q_start)
            float_assign_2d(ctx.model.joint_qd, ctx.joint_qd_start)
            # updates body position/vel
            for _ in range(substeps):
                state_out = model.state(requires_grad=True)
                state_in = integrator.simulate(
                    model, state_in, state_out, dt / float(substeps)
                )
            ctx.state_out = state_in
            # updates joint_q joint_qd
            ctx.joint_q_end, ctx.joint_qd_end = model.joint_q, model.joint_qd
            wp.sim.eval_ik(ctx.model, ctx.state_out, ctx.joint_q_end, ctx.joint_qd_end)

        ctx.outputs = to_weak_list(
            ctx.state_out.flatten() + [ctx.joint_q_end, ctx.joint_qd_end]
        )
        return tuple([wp.to_torch(x) for x in ctx.outputs])

    @staticmethod
    def backward(ctx, *adj_outputs):  # , adj_joint_q, adj_joint_qd):
        for adj_out, out in zip(adj_outputs, ctx.outputs):
            out.grad = wp.from_torch(adj_out)
        ctx.tape.backward()
        adj_inputs = [ctx.tape.get(x, None) for x in self.inputs]
        return (None, None, None, None, None, *filter_grads(adj_inputs))


class IntegratorSimulate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        act,
        joint_q_start,
        joint_qd_start,
        joint_q_end,
        joint_qd_end,
        model,
        state_in,
        integrator,
        sim_dt,
    ):
        ctx.tape = wp.Tape()
        ctx.model = model
        ctx.act = wp.from_torch(act)
        ctx.joint_q_start = wp.from_torch(joint_q_start)
        ctx.joint_qd_start = wp.from_torch(joint_qd_start)
        ctx.joint_q_end = wp.from_torch(joint_q_end)
        ctx.joint_qd_end = wp.from_torch(joint_qd_end)
        ctx.act.requires_grad = True
        # allocate output
        ctx.state_out = model.state(requires_grad=True)

        with ctx.tape:
            float_assign_joint_act(ctx.model.joint_act, ctx.act)
            float_assign_2d(ctx.model.joint_q, ctx.joint_q_start)
            float_assign_2d(ctx.model.joint_qd, ctx.joint_qd_start)
            # updates body position/vel
            ctx.state_out = integrator.simulate(model, state_in, ctx.state_out, sim_dt)
            # updates joint_q joint_qd
            float_assign(ctx.joint_q_end, ctx.model.joint_q)
            float_assign(ctx.joint_qd_end, ctx.model.joint_qd)
            wp.sim.eval_ik(ctx.model, ctx.state_out, ctx.joint_q_end, ctx.joint_qd_end)

        return (
            wp.to_torch(ctx.joint_q_end).view(-1, 2),
            wp.to_torch(ctx.joint_qd_end).view(-1, 2),
        )

    @staticmethod
    def backward(ctx, adj_joint_q, adj_joint_qd):

        # map incoming Torch grads to our output variables
        ctx.joint_q_end.grad = wp.from_torch(adj_joint_q.flatten())
        ctx.joint_qd_end.grad = wp.from_torch(adj_joint_qd.flatten())

        ctx.tape.backward()
        joint_act_grad = wp.to_torch(ctx.tape.gradients[ctx.act])
        joint_q_grad = wp.to_torch(ctx.tape.gradients[ctx.joint_q_start])
        joint_qd_grad = wp.to_torch(ctx.tape.gradients[ctx.joint_qd_start])
        # print(f"joint_q_grad.norm(): {joint_q_grad.norm()}")
        # print(f"joint_qd_grad.norm(): {joint_qd_grad.norm()}")
        # print(f"joint_act_grad.norm(): {joint_act_grad.norm()}")

        # return adjoint w.r.t. inputs
        return (
            joint_act_grad,
            joint_q_grad,
            joint_qd_grad,
            None,
            None,
            None,
            None,
            None,
            None,
        )
