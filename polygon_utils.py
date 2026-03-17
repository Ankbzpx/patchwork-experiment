from functools import partial

from common import normalize

import jax
from jax import jit, numpy as jnp, vmap
import numpy as np

from icecream import ic


def regular_square():
    return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])


def regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius


def star(n_points=5, outer_radius=1.0, inner_radius=0.5):
    angles = np.linspace(0, 2 * np.pi, n_points * 2, endpoint=False)
    radii = np.empty(n_points * 2)
    radii[0::2] = outer_radius
    radii[1::2] = inner_radius
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


def gen_edge(pt):
    N = len(pt)
    return np.stack([np.roll(np.arange(N), -1), np.arange(N)], -1)


@jit
def edge_normal(V_edge):
    v0 = V_edge[0]
    v1 = V_edge[1]

    return jnp.array([v0[1] - v1[1], v1[0] - v0[0]])


@partial(jit, static_argnames=("n"))
def sample_edge(V_edge, n):
    v0 = V_edge[0][None, :]
    v1 = V_edge[1][None, :]
    t = jnp.linspace(0, 1, n + 2)[1:-1, None]
    return v0 * (1 - t) + v1 * t


def sample_edge_uniform(V, E, n):
    per_edge_verts = V[E]
    V_up = vmap(sample_edge, in_axes=(0, None))(per_edge_verts, n)
    V_up = jnp.permute_dims(V_up, (1, 0, 2)).reshape(-1, 2)
    V_n = vmap(edge_normal)(per_edge_verts)
    V_up_n = jnp.repeat(V_n[:, None, :], n, axis=1)
    V_up_n = jnp.permute_dims(V_up_n, (1, 0, 2)).reshape(-1, 2)
    V_up_n = vmap(normalize)(V_up_n)
    return V_up, V_up_n


@partial(jit, static_argnames=("n"))
def sample_edge_interval(V_edge, key, n):
    v0 = V_edge[0][None, :]
    v1 = V_edge[1][None, :]
    t = jax.random.uniform(key, (n,))[:, None]
    return v0 * (1 - t) + v1 * t


def sample_edge_random(V, E, n, seed=0):
    per_edge_verts = V[E]
    N = len(per_edge_verts)
    keys = jax.random.split(jax.random.PRNGKey(seed), N)
    V_up = vmap(sample_edge_interval, in_axes=(0, 0, None))(
        per_edge_verts, keys, n
    ).reshape(-1, 2)
    V_n = vmap(edge_normal)(per_edge_verts)
    V_up_n = jnp.repeat(V_n[:, None, :], n, axis=1).reshape(-1, 2)
    V_up_n = vmap(normalize)(V_up_n)
    return V_up, V_up_n
