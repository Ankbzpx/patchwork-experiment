import argparse
import os

from common import CheckPointer, normalize, normalize_aabb
from dataset_utils import config_toy_dataloader
from patchwork import fit_patchwork
from polygon_utils import sample_edge_uniform
from rasterizer import PatchworkRasterizer

import equinox as eqx
import jax
from jax import jit, numpy as jnp, vmap
from jaxtyping import Array, Float, Int, PyTree
import matplotlib.pyplot as plt
import numpy as np

from icecream import ic
import polyscope as ps


@jit
def cosine_similarity(x, y):
    demo = jnp.linalg.norm(x) * jnp.linalg.norm(y)

    return jnp.dot(x, y) / jnp.where(demo > 1e-8, demo, 1e-8)


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


def init_plr(sur_samples, sur_normals, rho=100.0) -> Patchwork:
    samples_init = sur_samples
    sample_normals_init = sur_normals

    a = rho * samples_init + sample_normals_init
    b = rho * samples_init
    f = 0.5 * rho * np.einsum("ni,ni->n", samples_init, samples_init)
    c = f - np.einsum("ni,ni->n", a, samples_init)
    d = f - np.einsum("ni,ni->n", b, samples_init)

    return Patchwork(a, b, c, d)


if __name__ == "__main__":
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--exp_dir", type=str, help="Exp folder")
    parser.add_argument("--vis", action="store_true", help="Evaluate only")
    args = parser.parse_args()

    checkpointer = CheckPointer(exp_dir=args.exp_dir)

    model_name = args.model_name
    tag = f"{model_name}"

    N = 1024
    n_steps = 10000
    batch_size = 1024
    softplus_thr = -4.6

    polygon_path = os.path.join("data/polygon", f"{model_name}.npz")
    data = np.load(polygon_path)
    V = data["V"]
    E = data["E"]
    V = normalize_aabb(V)

    sur_samples, sur_normals = sample_edge_uniform(V, E, (batch_size // len(E)) + 1)
    mask = vmap(jnp.linalg.norm)(sur_normals) > 0
    sur_samples = sur_samples[mask]
    sur_normals = sur_normals[mask]
    sur_samples = sur_samples[:batch_size]
    sur_normals = sur_normals[:batch_size]

    dataloader = config_toy_dataloader(n_steps, batch_size)
    model, meta = fit_patchwork(
        sur_samples,
        sur_normals,
        dataloader,
        N=N,
        lr=1e-2,
        w_prune=1.0,
        softplus_thr=softplus_thr,
        n_steps=n_steps,
        batch_size=batch_size,
    )
    checkpointer.serialize(model, tag, meta)

    mask1 = model.s1 > softplus_thr
    mask2 = model.s2 > softplus_thr
    a, c, s1, beta1, b, d, s2, beta2 = model.unpack_coeffs()
    a = a[mask1]
    c = c[mask1]
    s1 = s1[mask1]
    b = b[mask2]
    d = d[mask2]
    s2 = s2[mask2]
    A = jnp.vstack([jnp.hstack([a, c[:, None]]), jnp.hstack([b, d[:, None]])])
    s = np.concat([s1, -s2])

    rasterizer = PatchworkRasterizer(2048)
    coeffs = jnp.hstack([A, s[:, None]])
    mask = np.ones((len(coeffs),))
    line_scale = 0.5
    use_softmax = True
    draw_candidate = True
    pad_eps = False

    img = rasterizer.rasterize(
        coeffs,
        mask,
        0.5 * (beta1 + beta2).item(),
        line_scale,
        use_softmax,
        draw_candidate,
        pad_eps,
    )
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.savefig(
        f"{checkpointer.exp_dir}/{model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    if args.vis:
        plt.show()

    plt.close(fig)
