import argparse
import os

from common import normalize_aabb
from loss import logsumexp_init, normalize
from polygon_utils import sample_edge_random


from admm import rasterize_patchwork  # isort: skip

import equinox as eqx
import jax
from jax import jit, numpy as jnp, vmap
from jaxtyping import Array, Float, PyTree
import numpy as np
import optax

from icecream import ic
import polyscope as ps
import polyscope.imgui as psim


@jit
def cosine_similarity(x, y):
    demo = jnp.linalg.norm(x) * jnp.linalg.norm(y)
    return jnp.dot(x, y) / jnp.where(demo > 1e-8, demo, 1e-8)


class Patchwork(eqx.Module):
    ga: Array
    gb: Array
    da: Array
    db: Array
    s1: Array
    s2: Array

    def __init__(self, pc, pc_normal, N, offset=1e-2):

        center = pc[:N]
        dir = vmap(normalize)(pc_normal[:N])
        offset = jnp.array([offset])

        self.ga = center + offset * dir
        self.gb = center - offset * dir

        self.da = jnp.zeros((len(center),))
        self.db = jnp.zeros((len(center),))

        self.s1 = 2e-2 * jnp.ones((len(center),))
        self.s2 = 2e-2 * jnp.ones((len(center),))

    def unpack_coeffs(self):
        a = 2 * self.ga
        b = 2 * self.gb

        ra2 = jax.nn.softplus(1e-1 * self.da)
        rb2 = jax.nn.softplus(1e-1 * self.db)

        # For now, force nonnegative power
        c = -jnp.square(self.ga).sum(1) + ra2
        d = -jnp.square(self.gb).sum(1) + rb2

        s1 = jax.nn.sigmoid(1e2 * self.s1)
        s2 = jax.nn.sigmoid(1e2 * self.s2)

        return a, c, s1, b, d, s2, ra2, rb2


def ps_draw_patchwork() -> callable:
    V = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    F = np.array([[0, 1, 2], [1, 3, 2]])
    Vt = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    ps_mesh = ps.register_surface_mesh("Fullscreen", V, F)
    ps_mesh.set_material("flat")
    ps_mesh.add_parameterization_quantity("Vt", Vt, defined_on="vertices", enabled=True)

    def _draw(a, c, s1, b, d, s2, ra2, rb2):
        A = jnp.vstack([jnp.hstack([a, c[:, None]]), jnp.hstack([b, d[:, None]])])
        s = jnp.concat(
            [jnp.where(s1 > 0.5, 1.0, 1e-8), -jnp.where(s2 > 0.5, 1.0, 1e-8)]
        )

        ps_mesh.add_color_quantity(
            "patchwork",
            np.transpose(rasterize_patchwork(A, s, line_scale=0.25), (1, 0, 2))[
                ::-1, :, :
            ],
            defined_on="texture",
            param_name="Vt",
            enabled=True,
        )

    return _draw


@jit
def symmetric_reparam(a, b):
    return (0.5 * (a + b) + 0.5 * jax.lax.stop_gradient(a - b)) - (
        0.5 * (a + b) - 0.5 * jax.lax.stop_gradient(a - b)
    )


@jit
def anti_symmetric_reparam(a, b):
    return (0.5 * jax.lax.stop_gradient(a + b) + 0.5 * (a - b)) - (
        0.5 * jax.lax.stop_gradient(a + b) - 0.5 * (a - b)
    )


@jit
def eval_mse_res(X, ga, gb, ra2, rb2):
    return jnp.abs(
        jnp.einsum(
            "nd,nd->n",
            (ga + gb) - 2 * X,
            jax.lax.stop_gradient(ga - gb),
        )
        - (ra2 - rb2)
    )


@jit
def eval_normal_res(XN, ga, gb):
    return vmap(cosine_similarity)(
        ga - gb,
        XN,
    )


@jit
def eval_cover_res(X, ga, ga_sec, ra2, ra2_sec):
    return jax.nn.relu(
        jnp.einsum(
            "nd,nd->n",
            (ga_sec + ga) - 2 * X,
            (ga_sec - ga),
        )
        - (ra2_sec - ra2)
    )


def main():
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--exp_dir", type=str, help="Exp folder")
    parser.add_argument("--vis", action="store_true", help="Evaluate only")
    args = parser.parse_args()

    model_name = args.model_name
    tag = f"{model_name}"

    polygon_path = os.path.join("data/polygon", f"{model_name}.npz")
    data = np.load(polygon_path)
    V = data["V"]
    E = data["E"]
    E = E[:, ::-1]
    V = normalize_aabb(V)

    # debug_init(V, E)
    # exit()

    @jit
    def sdf_2d(pt):
        a = V[E[:, 0]]
        b = V[E[:, 1]]

        pa = a - pt
        pb = b - pt
        ba = a - b

        cp = jnp.cross(pa, pb)
        dp = (pa * pb).sum(1)
        signed_angle = jnp.atan2(cp, dp)
        # Winding number
        wn = signed_angle.sum() / (2 * jnp.pi)

        t = jnp.clip((pa * ba).sum(1) / (ba * ba).sum(1), 0.0, 1.0)
        p_closest = a + t[:, None] * (b - a)
        dist = jnp.linalg.norm(pt - p_closest, axis=1).min()

        return jnp.where(jnp.abs(wn) > 0.5, -dist, dist)

    k = 2
    sur_samples, sur_normals = sample_edge_random(V, E, k)

    N = len(sur_samples)
    D = 2

    model = Patchwork(sur_samples, sur_normals, N)
    batch_size = len(E) * k
    logsumexp = logsumexp_init(N, batch_size, D)

    optim = optax.chain(
        optax.scale_by_adam(),
        optax.clip_by_global_norm(5),
        optax.scale_by_learning_rate(
            optax.schedules.cosine_decay_schedule(3e-4, 10000)
        ),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    data_key = jax.random.PRNGKey(0)
    is_fitting = False
    n_iter = 0

    @jit
    def _eval(model: Patchwork, X: Array):
        a, c, s1, b, d, s2, ra2, rb2 = model.unpack_coeffs()

        A = jnp.hstack([a, c[:, None]])
        B = jnp.hstack([b, d[:, None]])

        X = jnp.hstack([X, jnp.ones((len(X), 1))])
        A = jnp.hstack([a, c[:, None]])
        B = jnp.hstack([b, d[:, None]])

        _, _, h_idx, h_sec_idx = logsumexp(
            X,
            A,
            jnp.where(s1 > 0.5, 1.0, 1e-8),
            jnp.ones((1,)),
        )
        _, _, g_idx, g_sec_idx = logsumexp(
            X,
            B,
            jnp.where(s2 > 0.5, 1.0, 1e-8),
            jnp.ones((1,)),
        )

        return (
            model.ga,
            model.gb,
            s1,
            s2,
            ra2,
            rb2,
            h_idx,
            h_sec_idx,
            g_idx,
            g_sec_idx,
        )

    @eqx.filter_jit
    def loss_func(
        model: Patchwork,
        sur_samples: Array,
        sur_normals: Array,
        samples: Array,
        sample_sdfs: Array,
        s1_mask: Array,
        s2_mask: Array,
    ):
        ga, gb, s1, s2, ra2, rb2, h_idx, h_sec_idx, g_idx, g_sec_idx = _eval(
            model, sur_samples
        )

        ra2_h = ra2[h_idx]
        rb2_g = rb2[g_idx]

        ra2_h_sec = ra2[h_sec_idx]
        rb2_g_sec = rb2[g_sec_idx]

        ga_h = ga[h_idx]
        gb_g = gb[g_idx]

        ga_h_sec = ga[h_sec_idx]
        gb_g_sec = gb[g_sec_idx]

        candidate_normals = jnp.stack(
            [
                ga_h - gb_g,
                ga_h - gb_g_sec,
                ga_h_sec - gb_g,
                ga_h_sec - gb_g_sec,
            ],
            axis=1,
        )

        candidate_mse_res = jnp.stack(
            [
                eval_mse_res(sur_samples, ga_h, gb_g, ra2_h, rb2_g),
                eval_mse_res(sur_samples, ga_h, gb_g_sec, ra2_h, rb2_g_sec),
                eval_mse_res(sur_samples, ga_h_sec, gb_g, ra2_h_sec, rb2_g),
                eval_mse_res(sur_samples, ga_h_sec, gb_g_sec, ra2_h_sec, rb2_g_sec),
            ],
            axis=1,
        )

        candidate_normal_res = jnp.stack(
            [
                eval_normal_res(sur_normals, ga_h, gb_g),
                eval_normal_res(sur_normals, ga_h, gb_g_sec),
                eval_normal_res(sur_normals, ga_h_sec, gb_g),
                eval_normal_res(sur_normals, ga_h_sec, gb_g_sec),
            ],
            axis=1,
        )

        candidate_cover_res = jnp.stack(
            [
                jnp.zeros_like(ra2_h),
                eval_cover_res(
                    sur_samples,
                    gb_g,
                    gb_g_sec,
                    rb2_g,
                    rb2_g_sec,
                ),
                eval_cover_res(
                    sur_samples,
                    ga_h,
                    ga_h_sec,
                    ra2_h,
                    ra2_h_sec,
                ),
                eval_cover_res(
                    sur_samples,
                    ga_h,
                    ga_h_sec,
                    ra2_h,
                    ra2_h_sec,
                )
                + eval_cover_res(
                    sur_samples,
                    gb_g,
                    gb_g_sec,
                    rb2_g,
                    rb2_g_sec,
                ),
            ],
            axis=1,
        )

        dp = jnp.einsum(
            "bnd,bd->bn", vmap(vmap(normalize))(candidate_normals), sur_normals
        )

        # Hard assignment seems better than Softmax with small temperature
        pick = jnp.argmax(dp, axis=1)
        dp_pick = jnp.take_along_axis(dp, pick[:, None], axis=1)[:, 0]
        d_theta = jnp.arccos(jnp.clip(dp[:, 0], -1, 1)) - jnp.arccos(
            jnp.clip(dp_pick, -1, 1)
        )
        # Keep the current assignment if they are close enough
        pick = jnp.where(jnp.abs(d_theta) > jnp.deg2rad(5.0), pick, 0)

        loss_mse = jnp.take_along_axis(candidate_mse_res, pick[:, None], axis=1).mean()
        norm_res_pick = jnp.take_along_axis(
            candidate_normal_res, pick[:, None], axis=1
        )[:, 0]
        loss_normal = (1 - jnp.where(norm_res_pick > 0.0, norm_res_pick, 0.0)).mean()
        loss_cover = jnp.take_along_axis(
            candidate_cover_res, pick[:, None], axis=1
        ).mean()

        loss = 10 * loss_mse + loss_normal + 100 * loss_cover

        # %%
        h_idx_candidate = jnp.stack([h_idx, h_idx, h_sec_idx, h_sec_idx], axis=1)
        g_idx_candidate = jnp.stack([g_idx, g_sec_idx, g_idx, g_sec_idx], axis=1)

        s1_mask = s1_mask.at[
            h_idx_candidate.reshape(
                -1,
            )
        ].set(0)
        s1_mask = s1_mask.at[
            jnp.take_along_axis(
                h_idx_candidate,
                pick[:, None],
                axis=1,
            )[:, 0]
        ].set(1)

        s2_mask = s2_mask.at[
            g_idx_candidate.reshape(
                -1,
            )
        ].set(0)
        s2_mask = s2_mask.at[
            jnp.take_along_axis(
                g_idx_candidate,
                pick[:, None],
                axis=1,
            )[:, 0]
        ].set(1)
        loss_prune = (
            jnp.where(s1_mask, jnp.abs(1 - s1), jnp.abs(s1)).mean()
            + jnp.where(s2_mask, jnp.abs(1 - s2), jnp.abs(s2)).mean()
        )
        loss += 0.1 * loss_prune

        loss_dict = {
            "loss": loss,
            "loss_mse": loss_mse,
            "loss_normal": loss_normal,
            "loss_cover": loss_cover,
            "h_idx": h_idx,
            "g_idx": g_idx,
            "h_sec_idx": h_sec_idx,
            "g_sec_idx": g_sec_idx,
            "s1_mask": s1_mask,
            "s2_mask": s2_mask,
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

    s1_mask = jnp.ones_like(model.s1)
    s2_mask = jnp.ones_like(model.s2)

    def callback():
        nonlocal data_key, model, opt_state, is_fitting, n_iter, s1_mask, s2_mask

        psim.Text(f"n_iter: {n_iter}")

        _, is_fitting = psim.Checkbox("Fit", is_fitting)
        if is_fitting:
            n_iter += 1

            data_key, _ = jax.random.split(data_key)
            sur_samples, sur_normals = sample_edge_random(V, E, k, data_key)
            samples = jax.random.uniform(
                data_key, (batch_size, 2), minval=-1.1, maxval=1.1
            )
            sample_sdfs = vmap(sdf_2d)(samples)
            batch = {
                "sur_samples": sur_samples,
                "sur_normals": sur_normals,
                "samples": samples,
                "sample_sdfs": sample_sdfs,
                "s1_mask": s1_mask,
                "s2_mask": s2_mask,
            }

            model, opt_state, loss_dict = make_step(model, opt_state, **batch)

            h_idx = loss_dict["h_idx"]
            g_idx = loss_dict["g_idx"]
            h_sec_idx = loss_dict["h_sec_idx"]
            g_sec_idx = loss_dict["g_sec_idx"]

            s1_mask = loss_dict["s1_mask"]
            s2_mask = loss_dict["s2_mask"]

            idx = jnp.arange(batch_size)
            E_h = jnp.stack([idx, batch_size + h_idx], axis=-1)
            E_g = jnp.stack([idx, batch_size + g_idx], axis=-1)
            E_h_sec = jnp.stack([idx, batch_size + h_sec_idx], axis=-1)
            E_g_sec = jnp.stack([idx, batch_size + g_sec_idx], axis=-1)

            V_a = jnp.vstack([sur_samples, model.ga])
            V_b = jnp.vstack([sur_samples, model.gb])

            ga_color = jnp.zeros((N, 3))
            ga_color = ga_color.at[:, 0].set(jax.nn.sigmoid(10.0 * model.s1))
            gb_color = jnp.zeros((N, 3))
            gb_color = gb_color.at[:, 2].set(jax.nn.sigmoid(10.0 * model.s2))

            ps_ga.update_point_positions(model.ga)
            ps_ga.add_color_quantity("s1", ga_color)
            ps_gb.update_point_positions(model.gb)
            ps_gb.add_color_quantity("s2", gb_color)
            draw_func(*model.unpack_coeffs())

            ps.register_curve_network("C_h", V_a, E_h, radius=2e-4)
            ps.register_curve_network("C_g", V_b, E_g, radius=2e-4)
            ps.register_curve_network("C_h_sec", V_a, E_h_sec, radius=2e-4)
            ps.register_curve_network("C_g_sec", V_b, E_g_sec, radius=2e-4)

    ga_color = jnp.zeros((N, 3))
    ga_color = ga_color.at[:, 0].set(1)
    gb_color = jnp.zeros((N, 3))
    gb_color = gb_color.at[:, 2].set(1)

    ps.init()
    ps.set_up_dir("neg_y_up")
    ps.set_ground_plane_mode("shadow_only")
    draw_func = ps_draw_patchwork()
    draw_func(*model.unpack_coeffs())
    ps.set_user_callback(callback)
    ps_ga = ps.register_point_cloud("ga", model.ga, radius=2e-3)
    ps_ga.add_color_quantity("s1", ga_color, enabled=True)
    ps_gb = ps.register_point_cloud("gb", model.gb, radius=2e-3)
    ps_gb.add_color_quantity("s2", gb_color, enabled=True)
    ps.register_curve_network("polygon", V, E, radius=2e-3)
    ps.set_automatically_compute_scene_extents(False)
    ps.show()

    a, c, s1, b, d, s2, _, _ = model.unpack_coeffs()
    A = jnp.vstack([jnp.hstack([a, c[:, None]]), jnp.hstack([b, d[:, None]])])
    s = jnp.concat([s1, -s2])
    ic(jnp.hstack([A, s[:, None]]))

    save_path = os.path.join(args.exp_dir, f"{tag}.npz")
    ic(A.shape, s.shape)
    os.makedirs(args.exp_dir, exist_ok=True)
    np.savez(save_path, A=A, s=s)


if __name__ == "__main__":
    main()
