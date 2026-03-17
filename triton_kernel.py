from typing import List

from jaxtyping import Array, Float, Int
import torch
import triton
import triton.language as tl

from icecream import ic


def get_cuda_autotune_config():
    return (
        [
            triton.Config(
                {
                    "BLOCK_SIZE_N": BLOCK_SIZE_N,
                    "BLOCK_SIZE_M": int(8192 / BLOCK_SIZE_N),
                },
                num_warps=num_warps,
                num_stages=num_stages,
            )
            for BLOCK_SIZE_N in [16, 32, 64]
            for num_warps in [2, 4, 8]
            for num_stages in [1, 2, 3]
        ]
        + [
            triton.Config(
                {
                    "BLOCK_SIZE_N": int(8192 / BLOCK_SIZE_M),
                    "BLOCK_SIZE_M": BLOCK_SIZE_M,
                },
                num_warps=num_warps,
                num_stages=num_stages,
            )
            for BLOCK_SIZE_M in [16, 32, 64]
            for num_warps in [2, 4, 8]
            for num_stages in [1, 2, 3]
        ]
        + [
            triton.Config(
                {
                    "BLOCK_SIZE_N": BLOCK_SIZE_N,
                    "BLOCK_SIZE_M": int(4096 / BLOCK_SIZE_N),
                },
                num_warps=num_warps,
                num_stages=num_stages,
            )
            for BLOCK_SIZE_N in [16, 32, 64]
            for num_warps in [2, 4, 8]
            for num_stages in [1, 2, 3]
        ]
        + [
            triton.Config(
                {
                    "BLOCK_SIZE_N": int(4096 / BLOCK_SIZE_M),
                    "BLOCK_SIZE_M": BLOCK_SIZE_M,
                },
                num_warps=num_warps,
                num_stages=num_stages,
            )
            for BLOCK_SIZE_M in [16, 32, 64]
            for num_warps in [2, 4, 8]
            for num_stages in [1, 2, 3]
        ]
    )


# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=["N", "M"],
# )
@triton.jit
def sum_exp_kernel(
    X_ptr,
    A_ptr,
    s_ptr,
    beta_ptr,
    z_max_ptr,
    z_max_idx_ptr,
    sum_exp_ptr,
    ws_ptr,
    N,
    M,
    D,
    X_stride_n,
    X_stride_d,
    A_stride_m,
    A_stride_d,
    s_stride_m,
    z_max_stride_n,
    z_max_idx_stride_n,
    sum_exp_stride_n,
    ws_stride_n,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    use_tf32: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    beta = tl.load(beta_ptr)

    # More L2 cache friendly launch (optional)
    num_pid_n = tl.num_programs(axis=0)
    num_pid_m = tl.num_programs(axis=1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_SIZE)

    base_n = pid_n * BLOCK_SIZE_N

    batch_base_n = base_n + tl.arange(0, BLOCK_SIZE_N)
    batch_n_mask = batch_base_n < N

    batch_base_d = tl.arange(0, 16)
    batch_d_mask = batch_base_d < D

    X = tl.load(
        X_ptr
        + (batch_base_n[:, None] * X_stride_n)
        + (batch_base_d[None, :] * X_stride_d),
        mask=batch_n_mask[:, None] & batch_d_mask[None, :],
        other=0.0,
    )
    if use_tf32:
        # Reference: https://github.com/triton-lang/triton/issues/4574
        ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
        X = tl.inline_asm_elementwise(
            ASM, "=r, r", [X], dtype=tl.float32, is_pure=True, pack=1
        )

    z_max = tl.full((BLOCK_SIZE_N,), -float("inf"), tl.float32)
    z_max_idx = tl.full((BLOCK_SIZE_N,), -1, tl.int32)
    sum_exp = tl.full((BLOCK_SIZE_N,), 0, tl.float32)
    ws = tl.full((BLOCK_SIZE_N,), 0, tl.float32)

    for base_m in tl.range(0, M, BLOCK_SIZE_M):
        batch_base_m = base_m + tl.arange(0, BLOCK_SIZE_M)
        batch_m_mask = batch_base_m < M

        A = tl.load(
            A_ptr
            + (batch_base_m[:, None] * A_stride_m)
            + (batch_base_d[None, :] * A_stride_d),
            mask=batch_m_mask[:, None] & batch_d_mask[None, :],
            other=0.0,
        )
        if use_tf32:
            A = tl.inline_asm_elementwise(
                ASM, "=r, r", [A], dtype=tl.float32, is_pure=True, pack=1
            )
        s = tl.load(s_ptr + batch_base_m * s_stride_m, mask=batch_m_mask)

        dp = tl.dot(X, A.T, input_precision="tf32" if use_tf32 else "ieee")
        z = beta * dp + tl.log(s)[None, :]
        z = tl.where(batch_n_mask[:, None] & batch_m_mask[None, :], z, -float("inf"))

        block_z_max, block_z_max_idx = tl.max(z, axis=1, return_indices=True)
        block_z_max_idx += base_m
        block_max_mask = block_z_max > z_max
        new_z_max = tl.where(block_max_mask, block_z_max, z_max)
        new_z_max_idx = tl.where(block_max_mask, block_z_max_idx, z_max_idx)

        block_sum_exp = tl.sum(tl.exp(z - block_z_max[:, None]), axis=1)
        block_ws = tl.sum(tl.exp(z - block_z_max[:, None]) * s[None, :], axis=1)
        sum_exp = (
            tl.exp(block_z_max - new_z_max) * block_sum_exp
            + tl.exp(z_max - new_z_max) * sum_exp
        )
        ws = tl.exp(block_z_max - new_z_max) * block_ws + tl.exp(z_max - new_z_max) * ws
        z_max = new_z_max
        z_max_idx = new_z_max_idx

    tl.store(z_max_ptr + batch_base_n * z_max_stride_n, z_max, mask=batch_n_mask)
    tl.store(
        z_max_idx_ptr + batch_base_n * z_max_idx_stride_n, z_max_idx, mask=batch_n_mask
    )
    tl.store(sum_exp_ptr + batch_base_n * sum_exp_stride_n, sum_exp, mask=batch_n_mask)
    tl.store(ws_ptr + batch_base_n * ws_stride_n, ws / sum_exp, mask=batch_n_mask)


def eval_sum_exp_triton(
    X: Array,
    A: Array,
    s: Array,
    beta: Array,
    block_size_n: Int = 64,
    block_size_m: Int = 64,
    use_tf32: bool = False,
):
    assert A.is_contiguous(), "Matrix A must be contiguous"
    assert X.is_contiguous(), "Matrix samples must be contiguous"

    N, D = X.shape
    M = A.shape[0]
    z_max = torch.zeros((N,), device=X.device, dtype=X.dtype)
    z_max_idx = torch.zeros((N,), device=X.device, dtype=torch.int32)
    sum_exp = torch.zeros((N,), device=X.device, dtype=X.dtype)
    ws = torch.zeros((N,), device=X.device, dtype=X.dtype)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]), 1)

    configs = {
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_M": block_size_m,
        "GROUP_SIZE": 16,
        "num_warps": 2,
        "num_stages": 3,
        "use_tf32": use_tf32,
    }

    sum_exp_kernel[grid](
        X,
        A,
        s,
        beta,
        z_max,
        z_max_idx,
        sum_exp,
        ws,
        N,
        M,
        D,
        X.stride(0),
        X.stride(1),
        A.stride(0),
        A.stride(1),
        s.stride(0),
        z_max.stride(0),
        z_max_idx.stride(0),
        sum_exp.stride(0),
        ws.stride(0),
        **configs,
    )
    return z_max, ws, z_max_idx, sum_exp


# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=["N", "M"],
# )
@triton.jit
def grad_logsumexp_kernel(
    A_ptr,
    s_ptr,
    beta_ptr,
    X_ptr,
    z_max_ptr,
    sum_exp_ptr,
    dz_ptr,
    ds_ptr,
    grad_A_ptr,
    grad_s_ptr,
    grad_beta_ptr,
    N,
    M,
    D,
    A_stride_n,
    A_stride_d,
    s_stride_n,
    X_stride_m,
    X_stride_d,
    z_max_stride_m,
    sum_exp_stride_m,
    dz_stride_m,
    ds_stride_m,
    grad_A_stride_n,
    grad_A_stride_d,
    grad_s_stride_n,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    use_tf32: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    beta = tl.load(beta_ptr)

    # More L2 cache friendly launch (optional)
    num_pid_n = tl.num_programs(axis=0)
    num_pid_m = tl.num_programs(axis=1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_SIZE)

    base_n = pid_n * BLOCK_SIZE_N

    batch_base_n = base_n + tl.arange(0, BLOCK_SIZE_N)
    batch_n_mask = batch_base_n < N

    batch_base_d = tl.arange(0, 16)
    batch_d_mask = batch_base_d < D

    A = tl.load(
        A_ptr
        + (batch_base_n[:, None] * A_stride_n)
        + (batch_base_d[None, :] * A_stride_d),
        mask=batch_n_mask[:, None] & batch_d_mask[None, :],
        other=0.0,
    )
    if use_tf32:
        ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
        A = tl.inline_asm_elementwise(
            ASM, "=r, r", [A], dtype=tl.float32, is_pure=True, pack=1
        )
    s = tl.load(s_ptr + batch_base_n * s_stride_n, mask=batch_n_mask)

    grad_A = tl.full((BLOCK_SIZE_N, 16), 0, tl.float32)
    grad_s = tl.full((BLOCK_SIZE_N,), 0, tl.float32)
    grad_beta = tl.full((), 0.0, tl.float32)
    for base_m in tl.range(0, M, BLOCK_SIZE_M):
        batch_base_m = base_m + tl.arange(0, BLOCK_SIZE_M)
        batch_m_mask = batch_base_m < M

        X = tl.load(
            X_ptr
            + (batch_base_m[:, None] * X_stride_m)
            + (batch_base_d[None, :] * X_stride_d),
            mask=batch_m_mask[:, None] & batch_d_mask[None, :],
            other=0.0,
        )
        if use_tf32:
            X = tl.inline_asm_elementwise(
                ASM, "=r, r", [X], dtype=tl.float32, is_pure=True, pack=1
            )
        z_max = tl.load(z_max_ptr + batch_base_m * z_max_stride_m, mask=batch_m_mask)
        sum_exp = tl.load(
            sum_exp_ptr + batch_base_m * sum_exp_stride_m, mask=batch_m_mask
        )
        dz = tl.load(dz_ptr + batch_base_m * dz_stride_m, mask=batch_m_mask)
        ds = tl.load(ds_ptr + batch_base_m * ds_stride_m, mask=batch_m_mask)

        dp = tl.dot(A, X.T, input_precision="tf32" if use_tf32 else "ieee")
        z = beta * dp + tl.log(s)[:, None] - z_max[None, :]
        p = tl.exp(z) / sum_exp[None, :] * dz[None, :]
        p = tl.where(batch_n_mask[:, None] & batch_m_mask[None, :], p, 0)

        if use_tf32:
            p = tl.inline_asm_elementwise(
                ASM, "=r, r", [p], dtype=tl.float32, is_pure=True, pack=1
            )
        grad_A += tl.dot(p, X, input_precision="tf32" if use_tf32 else "ieee")
        grad_s += tl.sum(p / beta / s[:, None], axis=1)

        grad_s += tl.sum(
            tl.where(
                batch_n_mask[:, None] & batch_m_mask[None, :],
                tl.exp(z) / sum_exp[None, :] * ds[None, :],
                0,
            ),
            axis=1,
        )

        # Product rule
        grad_beta += tl.sum(
            tl.where(
                batch_n_mask[:, None] & batch_m_mask[None, :],
                p * (dp / beta - ((tl.log(sum_exp) + z_max) / (beta * beta))[None, :]),
                0,
            )
        )

    tl.store(
        grad_A_ptr
        + (batch_base_n[:, None] * grad_A_stride_n)
        + (batch_base_d[None, :] * grad_A_stride_d),
        grad_A,
        mask=batch_n_mask[:, None] & batch_d_mask[None, :],
    )
    tl.store(grad_s_ptr + batch_base_n * grad_s_stride_n, grad_s, mask=batch_n_mask)
    tl.store(grad_beta_ptr + pid_n, grad_beta)


def eval_grad_logsumexp_triton(
    A: Array,
    s: Array,
    beta: Array,
    zmax: Array,
    sum_exp: Array,
    dz: Array,
    ds: Array,
    X: Array,
    block_size_n: Int = 64,
    block_size_m: Int = 64,
    use_tf32: bool = False,
):
    assert A.is_contiguous(), "Matrix A must be contiguous"
    assert X.is_contiguous(), "Matrix samples must be contiguous"

    N, D = A.shape
    M = X.shape[0]

    grad_A = torch.zeros_like(A)
    grad_s = torch.zeros_like(s)
    grad_beta = torch.zeros((triton.cdiv(N, block_size_n),), device=beta.device)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]), 1)

    configs = {
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_M": block_size_m,
        "GROUP_SIZE": 16,
        "num_warps": 4,
        "num_stages": 1,
        "use_tf32": use_tf32,
    }

    grad_logsumexp_kernel[grid](
        A,
        s,
        beta,
        X,
        zmax,
        sum_exp,
        dz,
        ds,
        grad_A,
        grad_s,
        grad_beta,
        N,
        M,
        D,
        A.stride(0),
        A.stride(1),
        s.stride(0),
        X.stride(0),
        X.stride(1),
        zmax.stride(0),
        sum_exp.stride(0),
        dz.stride(0),
        ds.stride(0),
        grad_A.stride(0),
        grad_A.stride(1),
        grad_s.stride(0),
        **configs,
    )
    return grad_A, grad_s, grad_beta.sum()[None]


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def eval_LSE_naive(X, A, c, s, beta):
    logits = (
        beta * (torch.einsum("nd,md->nm", A, X) + c[:, None]) + torch.log(s)[:, None]
    )
    logits_max, max_idx = torch.max(logits, dim=0)
    ws = (torch.softmax(logits, dim=0).detach() * s[:, None]).sum(0)
    return (
        (
            torch.log(
                (
                    torch.exp(beta * (torch.einsum("nd,md->nm", A, X) + c[:, None]))
                    * s[:, None]
                ).sum(dim=0)
            )
            / beta
        ),
        ws,
        max_idx.int(),
    )


def eval_LSE_triton(X, A, c, s, beta):
    z_max, ws, z_max_idx, sum_exp = eval_sum_exp_triton(X, A, c, s, beta)
    return (torch.log(sum_exp) + z_max) / beta, z_max_idx


@torch.library.custom_op("patchwork::logsumexp", mutates_args=())
def logsumexp(
    X: torch.Tensor,
    A: torch.Tensor,
    s: torch.Tensor,
    beta: torch.Tensor,
) -> List[torch.Tensor]:
    z_max, ws, z_max_idx, sum_exp = eval_sum_exp_triton(X, A, s, beta)
    return (
        (torch.log(sum_exp) + z_max) / beta,
        z_max,
        ws,
        z_max_idx,
        sum_exp,
    )


def logsumexp_setup_context(ctx, inputs, output):
    X, A, s, beta = inputs
    _, z_max, _, _, sum_exp = output
    ctx.save_for_backward(X, A, s, beta, z_max, sum_exp)


def logsumexp_backward(ctx, grad_out):
    X, A, s, beta, z_max, sum_exp = ctx.saved_tensors
    dz, _, ds, _, _ = grad_out
    grad_A, grad_s, grad_beta = eval_grad_logsumexp_triton(
        A, s, beta, z_max, sum_exp, dz, ds, X
    )
    return None, grad_A, grad_s, grad_beta


logsumexp.register_autograd(logsumexp_backward, setup_context=logsumexp_setup_context)

if __name__ == "__main__":
    torch.random.manual_seed(0)

    N = 16384
    M = 16384
    A = torch.randn((M, 3)).cuda().requires_grad_(True)
    X = torch.randn((N, 3)).cuda()
    s = torch.rand((M,)).cuda().requires_grad_(True)
    c = torch.rand((M,)).cuda().requires_grad_(True)
    beta = torch.rand((1,)).cuda().requires_grad_(True)

    def test_loss_triton(X, A, c, s, beta):
        A = torch.hstack([A, c[:, None]])
        X = torch.hstack([X, torch.ones((len(X), 1), dtype=X.dtype, device=X.device)])
        val, _, ws, _, _ = logsumexp(X, A, s, beta)
        return val.sum() + ws.sum()

    loss_triton = test_loss_triton(X, A, c, s, beta)
    d_A_triton = gradient(loss_triton, A)
    d_c_triton = gradient(loss_triton, c)
    d_s_triton = gradient(loss_triton, s)
    d_beta_triton = gradient(loss_triton, beta)

    def test_loss_torch(X, A, c, s, beta):
        val, ws, _ = eval_LSE_naive(X, A, c, s, beta)
        return val.sum() + ws.sum()

    loss_torch = test_loss_torch(X, A, c, s, beta)

    d_A_torch = gradient(loss_torch, A)
    d_c_torch = gradient(loss_torch, c)
    d_s_torch = gradient(loss_torch, s)
    d_beta_torch = gradient(loss_torch, beta)

    ic(d_A_triton[:10], d_A_torch[:10])
    ic(d_c_triton[:10], d_c_torch[:10])
    ic(d_s_triton[:10], d_s_torch[:10])

    print(
        torch.allclose(d_A_triton, d_A_torch),
        torch.allclose(d_c_triton, d_c_torch),
        torch.allclose(d_s_triton, d_s_torch),
        torch.allclose(d_beta_triton, d_beta_torch),
    )
