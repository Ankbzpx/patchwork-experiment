from common import normalize
from loss import cosine_similarity, double_well_potential, logsumexp_init

import equinox as eqx
import jax
from jax import jit, numpy as jnp, vmap
from jaxtyping import Array, Float, Int, PyTree
import numpy as np
import optax
from tqdm import tqdm

from icecream import ic


class Patchwork(eqx.Module):
    a_dir: Array
    a_norm: Array
    c1: Array
    s1: Array
    beta1: Array
    b_dir: Array
    b_norm: Array
    c2: Array
    s2: Array
    beta2: Array

    @classmethod
    def init_plr(cls, sur_samples, sur_normals, N, rho):
        samples_init = sur_samples[:N]
        sample_normals_init = sur_normals[:N]

        a = rho * samples_init + sample_normals_init
        b = rho * samples_init
        f = 0.5 * rho * np.einsum("ni,ni->n", samples_init, samples_init)
        c = f - np.einsum("ni,ni->n", a, samples_init)
        d = f - np.einsum("ni,ni->n", b, samples_init)
        return cls(a, b, c, d)

    def __init__(self, a, b, c, d, beta=75.0):
        a_norm = vmap(jnp.linalg.norm)(a)
        self.a_dir = a / (a_norm[:, None] + 1e-8)
        self.a_norm = a_norm
        self.c1 = c
        self.s1 = jnp.log(jnp.e - 1) * jnp.ones_like(c)
        self.beta1 = jnp.array([beta])

        b_norm = vmap(jnp.linalg.norm)(b)
        self.b_dir = b / (b_norm[:, None] + 1e-8)
        self.b_norm = b_norm
        self.c2 = d
        self.s2 = jnp.log(jnp.e - 1) * jnp.ones_like(d)
        self.beta2 = jnp.array([beta])

    def unpack_coeffs(self):
        a = vmap(normalize)(self.a_dir) * self.a_norm[:, None]
        c = self.c1
        s1 = jax.nn.softplus(self.s1)
        beta1 = jax.nn.softplus(self.beta1)

        b = vmap(normalize)(self.b_dir) * self.b_norm[:, None]
        d = self.c2
        s2 = jax.nn.softplus(self.s2)
        beta2 = jax.nn.softplus(self.beta2)
        return a, c, s1, beta1, b, d, s2, beta2

    def __call__(self, x):
        a, c, s1, beta1, b, d, s2, beta2 = self.unpack_coeffs()
        h = (
            jax.nn.logsumexp(
                beta1 * (jnp.einsum("bd,nd->bn", x, a) + c[None, :]), axis=1, b=s1
            )
            / beta1
        )
        g = (
            jax.nn.logsumexp(
                beta2 * (jnp.einsum("bd,nd->bn", x, b) + d[None, :]), axis=1, b=s2
            )
            / beta2
        )
        return h - g


def fit_patchwork(
    sur_samples,
    sur_normals,
    dataloader,
    N,
    rho=200.0,
    lr=1e-3,
    w_mse=1.0,
    w_normal=1.0,
    w_reg=1.0,
    w_prune=10.0,
    softplus_thr=-4.6,
    n_steps=10000,
    batch_size=16384,
    **kwargs,
) -> tuple[Patchwork, dict]:
    D = sur_samples.shape[1]
    model = Patchwork.init_plr(sur_samples, sur_normals, N, rho=rho)
    logsumexp = logsumexp_init(N, batch_size, D)

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @jit
    def _eval(model: Patchwork, X: Array):
        a, c, s1, beta1, b, d, s2, beta2 = model.unpack_coeffs()

        X = jnp.hstack([X, jnp.ones((len(X), 1))])
        A = jnp.hstack([a, c[:, None]])
        B = jnp.hstack([b, d[:, None]])

        h, ws1, h_idx = logsumexp(X, A, s1, beta1)
        g, ws2, g_idx = logsumexp(X, B, s2, beta2)
        return h - g, a[h_idx] - b[g_idx], s1, s2, ws1, ws2

    @eqx.filter_jit
    def loss_func(model: Patchwork, samples: Array, sample_sdfs: Array):
        pred_sdfs_surf, pred_normals_surf, s1_surf, s2_surf, ws1_surf, ws2_surf = _eval(
            model, sur_samples
        )
        loss_mse = w_mse * jnp.abs(pred_sdfs_surf).mean()
        loss_normal = (
            w_normal
            * (1 - vmap(cosine_similarity)(pred_normals_surf, sur_normals)).mean()
        )
        pred_sdfs, _, s1, s2, ws1, ws2 = _eval(model, samples)

        pred_occs = jax.nn.sigmoid(-pred_sdfs)
        loss_reg = w_reg * jnp.square(double_well_potential(pred_occs)).mean()

        loss_prune = (
            w_prune * (s1_surf.mean() + s2_surf.mean() + s1.mean() + s2.mean())
            + jax.nn.relu(1 - ws1_surf).mean()
            + jax.nn.relu(1 - ws2_surf).mean()
            + jax.nn.relu(1 - ws1).mean()
            + jax.nn.relu(1 - ws2).mean()
        )

        # mse: Keep tessellation stiff
        # normal: Preserve visual look
        # reg: Expand cells, overlay region
        # prune: Remove elements
        loss = loss_mse + loss_normal + loss_reg + loss_prune
        loss_dict = {
            "loss_mse": loss_mse,
            "loss_normal": loss_normal,
            "loss_reg": loss_reg,
            "loss_prune": loss_prune,
        }
        return loss, loss_dict

    @eqx.filter_jit
    def make_step(model: Patchwork, opt_state: PyTree, **kwargs):
        grads, loss_dict = eqx.filter_grad(loss_func, has_aux=True)(model, **kwargs)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_dict

    pbar = tqdm(range(n_steps), dynamic_ncols=True)
    data_iter = iter(dataloader)

    for i in pbar:
        batch = next(data_iter)
        batch = jax.tree.map(lambda x: x.numpy()[0], batch)

        model, opt_state, loss_dict = make_step(model, opt_state, **batch)
        pbar.set_postfix(loss_dict)

        if i != 0 and (i % (n_steps // 5) - 1) == 0:
            model = eqx.tree_at(
                lambda x: x.s1,
                model,
                jnp.where(model.s1 < softplus_thr, -50, model.s1),
            )
            model = eqx.tree_at(
                lambda x: x.s2,
                model,
                jnp.where(model.s2 < softplus_thr, -50, model.s2),
            )

    num_params = (model.s1 > softplus_thr).sum() + (model.s2 > softplus_thr).sum()
    meta = {
        "num_params": (D + 1) * num_params.item(),
        "elapsed": pbar.format_dict["elapsed"],
    }

    return model, meta
