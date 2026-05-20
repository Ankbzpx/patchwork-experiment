from functools import partial

import jax
from jax import jit, numpy as jnp, vmap
from jaxtyping import Array

from icecream import ic
import polyscope as ps


# Reference: https://www.shadertoy.com/view/NljfRG
@jit
def rot(theta) -> Array:
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    return jnp.array([[c, -s], [s, c]])


def koch(p, n):
    p = rot(jnp.pi / 3) @ jnp.abs(p) - jnp.array([0, 0.5])
    w = 0.866
    for i in range(n):
        p = jnp.array([jnp.abs(p[0]) - w, -p[1]]) @ rot(jnp.pi / 6)
        w /= 1.732
        p = p.at[0].add(w)
    d = jnp.sign(p[1]) * jnp.linalg.norm(
        jnp.array([p[0] - jnp.clip(p[0], -w, w), p[1]])
    )
    return d


# Reference: https://iquilezles.org/articles/distfunctions2d/
def equilateral_triangle(p, r):
    k = jnp.sqrt(3)
    p = jnp.array([jnp.abs(p[0]) - r, p[1] + r / k])
    p = jax.lax.cond(
        p[0] + k * p[1] > 0,
        lambda x: 0.5 * jnp.array([x[0] - k * x[1], -k * x[0] - x[1]]),
        lambda x: x,
        p,
    )
    p = p.at[0].add(-jnp.clip(p[0], -2 * r, 0.0))
    return -jnp.sign(p[1]) * jnp.linalg.norm(p)


_koch_0 = jit(partial(equilateral_triangle, r=0.866))
_koch_2 = jit(partial(koch, n=2))
_koch_4 = jit(partial(koch, n=4))
_koch_6 = jit(partial(koch, n=6))
_koch_8 = jit(partial(koch, n=8))


@jit
def _scale_x(x):
    return x * 1.125


def koch_0(x):
    return _koch_0(_scale_x(x))


def koch_2(x):
    return _koch_2(_scale_x(x))


def koch_4(x):
    return _koch_4(_scale_x(x))


def koch_6(x):
    return _koch_6(_scale_x(x))


def koch_8(x):
    return _koch_8(_scale_x(x))


# Reference: https://robotic.tistory.com/5
@jit
def soft_min(x, y, eps=1e-6):
    return 0.5 * (x + y - jnp.sqrt((x - y) ** 2 + eps))


# sdf primitives
# Reference: https://iquilezles.org/articles/distfunctions2d/
@jit
def blobby_cross(pos, he=0.45):
    pos = jnp.abs(pos)
    pos = jnp.array([jnp.abs(pos[0] - pos[1]), 1.0 - pos[0] - pos[1]]) / jnp.sqrt(2)

    p = (he - pos[1] - 0.25 / he) / (6.0 * he)
    q = pos[0] / (he * he * 16)
    h = q * q - p * p * p

    r = jnp.where(h > 0.0, jnp.sqrt(h), jnp.sqrt(p))
    x = jnp.where(
        h > 0.0,
        jnp.power(q + r, 1.0 / 3.0)
        - jnp.power(jnp.abs(q - r), 1.0 / 3.0) * jnp.sign(r - q),
        2.0 * r * jnp.cos(jnp.arccos(q / (p * r)) / 3.0),
    )

    x = soft_min(x, jnp.sqrt(2.0) / 2.0)
    z = jnp.array([x, he * (1.0 - 2.0 * x * x)]) - pos
    return jnp.linalg.norm(z) * jnp.sign(z[1])


@jit
def hexgram(p, r=0.22):
    xy = jnp.array([-0.5, 0.8660254038])
    yx = jnp.array([0.8660254038, -0.5])
    z = 0.5773502692
    w = 1.7320508076

    p = jnp.abs(p)
    p = p - 2.0 * soft_min(jnp.dot(xy, p), 0.0) * xy
    p = p - 2.0 * soft_min(jnp.dot(yx, p), 0.0) * yx
    p = p - jnp.array([jnp.clip(p[0], r * z, r * w), r])
    return jnp.linalg.norm(p) * jnp.sign(p[1])


@jit
def equilateral_triangle(p, r=0.4):
    k = jnp.sqrt(3.0)

    p = p.at[0].set(jnp.abs(p[0]) - r)
    p = p.at[1].set(p[1] + r / k)
    p = jnp.where(
        (p[0] + k * p[1]) > 0.0, jnp.array([p[0] - k * p[1], -k * p[0] - p[1]]) / 2.0, p
    )
    p = p.at[0].add(-jnp.clip(p[0], -2.0 * r, 0.0))
    return -jnp.linalg.norm(p) * jnp.sign(p[1])


@jit
def rot2d(theta):
    return jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )


@jit
def circle(p, r=0.282):
    return jnp.linalg.norm(p) - r


# training sdfs
@jit
def star(x):
    return 0.5 * hexgram(2.0 * x, 0.6)


@jit
def two_stars(x):
    return soft_min(
        0.5 * hexgram(2.0 * x - 0.4, 0.6),
        0.25 * hexgram(rot2d(jnp.pi / 4) @ (2.0 + 4.0 * x), 0.6),
    )


@jit
def triangle(x):
    return 0.5 * equilateral_triangle(2.0 * x, 1.0)


def annular(x, p, sdf, d, c=0):
    return jnp.abs(sdf(x, p) + c) - d


@jit
def triangle_ring(x):
    return 0.5 * annular(2.0 * x, 1.5, equilateral_triangle, 0.25)


@jit
def triangle_ring_rot90(x):
    return 0.5 * annular(
        2.0 * rot2d(-1 * jnp.pi / 2) @ x, 1.5, equilateral_triangle, 0.25
    )


@jit
def clover(x):
    return 0.5 * (blobby_cross(2.0 * x, 1.2) - 0.5)


@jit
def clover_ring(x):
    return 0.5 * annular(2.0 * x, 1.2, blobby_cross, 0.15, -0.5)


@jit
def clover_ring_fat_rot90(x):
    return 0.5 * annular(2.0 * rot2d(jnp.pi / 4) @ x, 1.2, blobby_cross, 0.15, -0.5)


if __name__ == "__main__":
    grid_res = 256
    axis = jnp.linspace(-1, 1, grid_res)
    samples = jnp.stack(jnp.meshgrid(axis, axis), axis=-1).reshape(-1, 2)

    sdf = vmap(blobby_cross)(samples)

    ps.init()
    ps_viz = ps.register_point_cloud("samples", samples)
    ps_viz.add_scalar_quantity("sdf", sdf < 0, enabled=True)
    ps.show()
