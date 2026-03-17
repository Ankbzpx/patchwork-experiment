import argparse
from glob import glob
import os

from common import (
    CheckPointer,
    compute_metrics,
    extract_surface,
    normalize_aabb,
)
from dataset_utils import config_sdf_dataloader
from loss import logsumexp_init
from patchwork import fit_patchwork

import igl
from jax import jit, numpy as jnp, vmap
import numpy as np
import open3d as o3d
import trimesh

from icecream import ic
import polyscope as ps


if __name__ == "__main__":
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--exp_dir", type=str, help="Exp folder")
    parser.add_argument("--N", type=int, default=16384, help="Num samples for init")
    parser.add_argument("--rho", type=int, default=200, help="Rho for geo init")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--w_mse", type=float, default=1.0, help="Weight mse")
    parser.add_argument("--w_normal", type=float, default=1.0, help="Weight normal")
    parser.add_argument("--w_reg", type=float, default=1.0, help="Weight reg")
    parser.add_argument("--w_prune", type=float, default=10.0, help="Weight prune")
    parser.add_argument("--n_steps", type=int, default=10000, help="Num of iterations")
    parser.add_argument("--batch_size", type=int, default=16384, help="Batch size")
    parser.add_argument("--res", type=int, default=256, help="MC res")
    parser.add_argument("--debug", action="store_true", help="Debug metrics")
    parser.add_argument("--vis", action="store_true", help="Visualize results")
    args = parser.parse_args()

    checkpointer = CheckPointer(exp_dir=args.exp_dir)

    model_name = args.model_name
    tag = f"{model_name}"

    gt_path = glob(os.path.join("data/mesh", f"{model_name}.*"))[0]
    V_gt, F_gt = igl.read_triangle_mesh(gt_path)
    V_gt = normalize_aabb(V_gt)

    pc_path = f"data/pc/{model_name}.ply"
    pc_o3d = o3d.io.read_point_cloud(os.path.expandvars(pc_path))
    sur_samples = np.asarray(pc_o3d.points)
    sur_normals = np.asarray(pc_o3d.normals)

    dataloader = config_sdf_dataloader(
        f"data/sdf/{model_name}.npz", args.n_steps, args.batch_size
    )

    model, meta = fit_patchwork(sur_samples, sur_normals, dataloader, **vars(args))

    group_size = 16384
    logsumexp = logsumexp_init(args.N, group_size, 3)

    @jit
    def infer(X):
        a, c, s1, beta1, b, d, s2, beta2 = model.unpack_coeffs()

        X = jnp.hstack([X, jnp.ones((len(X), 1))])
        A = jnp.hstack([a, c[:, None]])
        B = jnp.hstack([b, d[:, None]])

        h, _, _ = logsumexp(X, A, s1, beta1)
        g, _, _ = logsumexp(X, B, s2, beta2)
        return h - g

    V, F, _ = extract_surface(infer, args.res, group_size=group_size, iso=0.0)
    save_path = os.path.join(checkpointer.exp_dir, "result_meshes", f"{tag}.obj")
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    igl.write_triangle_mesh(save_path, V, F)

    if args.debug:
        seed = 0
        sample_size = 1000000
        f1_percent = 1e-3

        gt_mesh: trimesh.Trimesh = trimesh.Trimesh(V_gt, F_gt)
        gt_samples, _ = trimesh.sample.sample_surface(gt_mesh, sample_size, seed=seed)
        aabb = gt_mesh.bounding_box.bounds
        max_bound = np.max(aabb[1] - aabb[0])
        f1_thr = f1_percent * max_bound

        result_mesh: trimesh.Trimesh = trimesh.Trimesh(V, F)
        result_samples, _ = trimesh.sample.sample_surface(
            result_mesh, sample_size, seed=seed
        )

        chamfer_dist, hausdorff_distance, f_1_score = compute_metrics(
            result_samples, gt_samples, f1_thr
        )
        meta["chamfer_dist"] = chamfer_dist
        meta["hausdorff_distance"] = hausdorff_distance
        meta["f_1_score"] = f_1_score

    checkpointer.serialize(model, tag, meta)

    if args.vis:
        ps.init()
        ps.register_surface_mesh("Mesh", V, F)
        ps.register_surface_mesh("GT", V_gt, F_gt)
        ps.show()
