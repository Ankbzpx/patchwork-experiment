import os


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

from functools import partial

from triton_kernel import eval_grad_logsumexp_triton, eval_sum_exp_triton

import jax
from jax import grad, jit, numpy as jnp, vmap
from jaxtyping import Array, Float, Int
from torch2jax import torch2jax


@jit
def double_well_potential(x):
    x = 2 * (x - 0.5)
    return jnp.square(x) - 2 * jnp.abs(x) + 1


@jit
def cosine_similarity(x, y):
    demo = jnp.linalg.norm(x) * jnp.linalg.norm(y)
    return jnp.dot(x, y) / jnp.where(demo > 1e-8, demo, 1e-8)


@jit
def eikonal(x, norm=1):
    return jnp.abs(jnp.linalg.norm(x) - norm)


# \frac{1}{\beta} \log \sum_i \exp(\beta (A_i \cdot X + c_i))
# Note that we normalize the range by \frac{1}{\beta}, so the backward pass is exactly the softmax (with temperature)
def logsumexp_init(N: Int, M: Int, D: Int):
    sum_exp_jax = torch2jax(
        eval_sum_exp_triton,
        jax.ShapeDtypeStruct((M, D + 1), jnp.float32),
        jax.ShapeDtypeStruct((N, D + 1), jnp.float32),
        jax.ShapeDtypeStruct((N,), jnp.float32),
        jax.ShapeDtypeStruct((1,), jnp.float32),
        output_shapes=(
            jax.ShapeDtypeStruct((M,), jnp.float32),
            jax.ShapeDtypeStruct((M,), jnp.float32),
            jax.ShapeDtypeStruct((M,), jnp.int32),
            jax.ShapeDtypeStruct((M,), jnp.float32),
        ),
    )

    eval_grad_logsumexp_jax = torch2jax(
        eval_grad_logsumexp_triton,
        jax.ShapeDtypeStruct((N, D + 1), jnp.float32),
        jax.ShapeDtypeStruct((N,), jnp.float32),
        jax.ShapeDtypeStruct((1,), jnp.float32),
        jax.ShapeDtypeStruct((M,), jnp.float32),
        jax.ShapeDtypeStruct((M,), jnp.float32),
        jax.ShapeDtypeStruct((M,), jnp.float32),
        jax.ShapeDtypeStruct((M,), jnp.float32),
        jax.ShapeDtypeStruct((M, D + 1), jnp.float32),
        output_shapes=(
            jax.ShapeDtypeStruct((N, D + 1), jnp.float32),
            jax.ShapeDtypeStruct((N,), jnp.float32),
            jax.ShapeDtypeStruct((1,), jnp.float32),
        ),
    )

    @jit
    def logsumexp_impl(X: Array, A: Array, s: Array, beta: Array):
        z_max, ws, z_max_idx, sum_exp = sum_exp_jax(X, A, s, beta)
        primal = (jnp.log(sum_exp) + z_max) / beta, ws, z_max_idx
        res = X, A, s, beta, z_max, sum_exp
        return primal, res

    @jax.custom_vjp
    def logsumexp(X: Array, A: Array, s: Array, beta: Array):
        primal, _ = logsumexp_impl(X, A, s, beta)
        return primal

    @jit
    def logsumexp_fwd(X: Array, A: Array, s: Array, beta: Array):
        return logsumexp_impl(X, A, s, beta)

    @jit
    def logsumexp_bwd(res, grad_out):
        X, A, s, beta, z_max, sum_exp = res
        dz, ds, _ = grad_out
        grad_A, grad_s, grad_beta = eval_grad_logsumexp_jax(
            A, s, beta, z_max, sum_exp, dz, ds, X
        )
        return (None, grad_A, grad_s, grad_beta)

    logsumexp.defvjp(logsumexp_fwd, logsumexp_bwd)
    return jit(logsumexp)


@jit
def logsumexp_jax(x: Array, A: Array, c: Array, s: Array, beta: Array):
    z = beta * (jnp.einsum("nd,d->n", A, x) + c)
    w = jax.nn.softmax(z + jnp.log(s))
    w = jax.lax.stop_gradient(w)
    return (
        (jax.nn.logsumexp(z, b=s) / beta)[0],
        (s * w).sum(),
        jnp.argmax(z + jnp.log(s)),
    )


def debug_stats(func, *kwargs):
    compiled = jax.jit(func).lower(*kwargs).compile()
    print(compiled.memory_analysis())


if __name__ == "__main__":
    from icecream import ic

    res = 16
    N = res**3
    M = 10000
    D = 3

    logsumexp = logsumexp_init(N, M, D)

    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (M, D))
    key, _ = jax.random.split(key)
    A = jax.random.normal(key, (N, D))
    key, _ = jax.random.split(key)
    c = jax.random.normal(key, (N,))
    key, _ = jax.random.split(key)
    s = jax.random.uniform(key, (N,), maxval=0.1)
    key, _ = jax.random.split(key)
    beta = jax.random.uniform(key, (1,))

    def test_loss(X, A, c, s, beta):
        X = jnp.hstack([X, jnp.ones((len(X), 1))])
        A = jnp.hstack([A, c[:, None]])
        val, ws, idx = logsumexp(X, A, s, beta)
        return jnp.mean(val) + jnp.sum(ws)

    grad_A = grad(test_loss, argnums=1)(X, A, c, s, beta)
    ic(grad_A[:10])
    grad_c = grad(test_loss, argnums=2)(X, A, c, s, beta)
    ic(grad_c[:10])
    grad_s = grad(test_loss, argnums=3)(X, A, c, s, beta)
    ic(grad_s[:10])
    grad_beta = grad(test_loss, argnums=4)(X, A, c, s, beta)
    ic(grad_beta)

    def test_loss2(X, A, c, s, beta):
        val, ws, idx = vmap(logsumexp_jax, in_axes=(0, None, None, None, None))(
            X, A, c, s, beta
        )
        return jnp.mean(val) + jnp.sum(ws)

    grad_A_ref = grad(test_loss2, argnums=1)(X, A, c, s, beta)
    ic(grad_A[:10])
    grad_c_ref = grad(test_loss2, argnums=2)(X, A, c, s, beta)
    ic(grad_c_ref[:10])
    grad_s_ref = grad(test_loss2, argnums=3)(X, A, c, s, beta)
    ic(grad_s_ref[:10])
    grad_beta_ref = grad(test_loss2, argnums=4)(X, A, c, s, beta)
    ic(grad_beta_ref)

    debug_stats(
        logsumexp,
        jnp.hstack([X, jnp.ones((len(X), 1))]),
        jnp.hstack([A, c[:, None]]),
        s,
        beta,
    )
    debug_stats(
        vmap(logsumexp_jax, in_axes=(0, None, None, None, None)), X, A, c, s, beta
    )

    debug_stats(grad(test_loss, argnums=1), X, A, c, s, beta)
    debug_stats(
        grad(test_loss2, argnums=1),
        X,
        A,
        c,
        s,
        beta,
    )
