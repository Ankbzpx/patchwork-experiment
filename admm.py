import os


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

import jax
from jax import jit, numpy as jnp, vmap
from jaxtyping import Array, Float, PyTree
from joblib import Memory
import matplotlib.pyplot as plt
import optimistix as optx
from optimistix._misc import sum_squares
from tqdm import tqdm
import warp as wp

from icecream import ic


location = "__pycache__"
memory = Memory(location)

wp.config.quiet = True
wp.config.kernel_cache_dir = location


# Reference: https://github.com/Siahkamari/Piecewise-linear-regression/blob/master/Python/piecewise_linear_estimation.py
@memory.cache
def fit_admm(
    X: Array,
    y: Array,
    rho: float = 1.0,
    max_iter: int = 5000,
) -> PyTree:
    n_samples, dim = X.shape
    rho = rho / len(X)

    # Primal
    u = jnp.zeros((n_samples,))
    v = jnp.zeros((n_samples,))
    a = jnp.zeros((n_samples, dim))
    b = jnp.zeros((n_samples, dim))
    primal = {"u": u, "v": v, "a": a, "b": b}

    # Slack
    S = jnp.zeros((n_samples, n_samples))
    T = jnp.zeros((n_samples, n_samples))
    slack = {"S": S, "T": T}

    # Dual
    alpha = jnp.zeros((n_samples, n_samples))
    beta = jnp.zeros((n_samples, n_samples))
    dual = {"alpha": alpha, "beta": beta}

    @jit
    def eval_f(primal, y):
        u = primal["u"]
        v = primal["v"]
        return u - v - y

    @jit
    def eval_c1(primal, X):
        u = primal["u"]
        a = primal["a"]

        return (
            u[:, None]
            - u[None, :]
            - jnp.einsum("jd, ijd->ij", a, X[:, None, :] - X[None, :, :])
        )

    @jit
    def eval_c2(primal, X):
        v = primal["v"]
        b = primal["b"]

        return (
            v[:, None]
            - v[None, :]
            - jnp.einsum("jd, ijd->ij", b, X[:, None, :] - X[None, :, :])
        )

    @jit
    def eval_r1(primal, slack, X):
        c1 = eval_c1(primal, X)
        return c1 - slack["S"]

    @jit
    def eval_r2(primal, slack, X):
        c2 = eval_c2(primal, X)
        return c2 - slack["T"]

    @jit
    def eval_objective(primal, slack, dual):
        f = eval_f(primal, y)
        h1 = eval_r1(primal, slack, X) + dual["alpha"]
        h2 = eval_r2(primal, slack, X) + dual["beta"]
        return 0.5 * sum_squares(f) + 0.5 * rho * (sum_squares(h1) + sum_squares(h2))

    @jit
    def update_slack(primal, dual):
        c1 = eval_c1(primal, X)
        c2 = eval_c2(primal, X)
        return {
            "S": jnp.maximum(0, c1 + dual["alpha"]),
            "T": jnp.maximum(0, c2 + dual["beta"]),
        }

    @jit
    def update_dual(primal, slack, dual):
        h1 = eval_r1(primal, slack, X) + dual["alpha"]
        h2 = eval_r2(primal, slack, X) + dual["beta"]
        return {"alpha": h1, "beta": h2}

    _, eval_B1 = jax.linearize(lambda z: eval_r1(primal, z, X), slack)
    _, eval_At1 = jax.vjp(lambda x: eval_r1(x, slack, X), primal)

    _, eval_B2 = jax.linearize(lambda z: eval_r2(primal, z, X), slack)
    _, eval_At2 = jax.vjp(lambda x: eval_r2(x, slack, X), primal)

    @jit
    def eval_dr1(d_slack):
        return jnp.sqrt(
            sum(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(
                        lambda x: sum_squares(x), eval_At1(eval_B1(d_slack))
                    )
                )
            )
        )

    @jit
    def eval_dr2(d_slack):
        return jnp.sqrt(
            sum(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(
                        lambda x: sum_squares(x), eval_At2(eval_B2(d_slack))
                    )
                )
            )
        )

    @jit
    def _eval_objective(primal, args):
        slack, dual = args
        return eval_objective(primal, slack, dual)

    solver = optx.NonlinearCG(rtol=1e-5, atol=1e-5)

    @jit
    def step(primal, slack, dual):
        sol = optx.minimise(
            _eval_objective, solver, primal, args=(slack, dual), max_steps=5000
        )
        primal = sol.value

        slack_old = slack
        slack = update_slack(primal, dual)
        d_slack = jax.tree_util.tree_map(lambda x, y: x - y, slack, slack_old)

        pr1 = jnp.linalg.norm(eval_r1(primal, slack, X))
        pr2 = jnp.linalg.norm(eval_r2(primal, slack, X))
        # jax.debug.print("Primal residual: {pr1} {pr2}", pr1=pr1, pr2=pr2)

        dr1 = rho * eval_dr1(d_slack)
        dr2 = rho * eval_dr2(d_slack)
        # jax.debug.print("Dual residual: {dr1} {dr2}", dr1=dr1, dr2=dr2)
        # jax.debug.print("Ratio: {r1} {r2}", r1=pr1 / dr1, r2=pr2 / dr2)

        dual = update_dual(primal, slack, dual)
        return (
            primal,
            slack,
            dual,
            {"ratio1": pr1 / (dr1 + 1e-7), "ratio2": pr2 / (dr2 + 1e-7)},
        )

    pbar = tqdm(range(max_iter), dynamic_ncols=True)
    for i in pbar:
        primal, slack, dual, stats = step(primal, slack, dual)
        pbar.set_postfix(stats)

    return primal


# Rasterize with kernel, bypass the uniform buffer restriction
@wp.kernel
def rasterize_patchwork(
    A: wp.array(dtype=wp.vec3f),
    s: wp.array(dtype=wp.float32),
    img: wp.array2d(dtype=wp.float32),
    H: int,
    W: int,
    line_width: float,
):
    idx_i, idx_j = wp.tid()

    uv = wp.vec3(
        2.0 * (wp.float32(idx_i) / wp.float32(H) - 0.5),
        2.0 * (wp.float32(idx_j) / wp.float32(W) - 0.5),
        1.0,
    )

    max_idx = wp.int32(0)
    second_max_idx = wp.int32(0)
    max_val = wp.float32(-1e20)
    second_max_val = wp.float32(-1e20)
    max_val_pos = wp.float32(-1e20)
    max_val_neg = wp.float32(-1e20)

    num_lines = A.shape[0]
    for i in range(num_lines):
        val = wp.dot(A[i], uv)
        if s[i] > 0:
            if max_val_pos < val:
                max_val_pos = val
        else:
            if max_val_neg < val:
                max_val_neg = val

        if max_val < val:
            second_max_val = max_val
            second_max_idx = max_idx
            max_val = val
            max_idx = i
        elif second_max_val < val:
            second_max_val = val
            second_max_idx = i

    coeff = A[max_idx] - A[second_max_idx]
    line_scale = wp.length(wp.vec2(coeff[0], coeff[1]))
    if abs(max_val - second_max_val) / line_scale < 0.01 * line_width:
        img[idx_i, idx_j] = 1.0
    elif max_val_pos - max_val_neg < 0:
        img[idx_i, idx_j] = 0.5
    else:
        img[idx_i, idx_j] = 0.0


if __name__ == "__main__":
    from sdf2d import clover_ring, koch_2, two_stars

    grid_res_train = 32
    axis = jnp.linspace(-1, 1, grid_res_train)
    X = jnp.stack(jnp.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    y = vmap(koch_2)(X)

    primal = fit_admm(X, y)
    a = primal["a"]
    b = primal["b"]
    u = primal["u"]
    v = primal["v"]
    c = u - (a * X).sum(-1)
    d = v - (b * X).sum(-1)

    A = jnp.vstack([jnp.hstack([a, c[:, None]]), jnp.hstack([b, d[:, None]])])
    s = jnp.concat([jnp.ones_like(y), -jnp.ones_like(y)])

    res = 1024
    img = wp.zeros((res, res), dtype=wp.float32)

    wp.launch(
        kernel=rasterize_patchwork,
        dim=(res, res),
        inputs=(
            wp.from_jax(A, dtype=wp.vec3),
            wp.from_jax(s, dtype=wp.float32),
            img,
            res,
            res,
            0.5,
        ),
    )

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(wp.to_jax(img))
    plt.show()
