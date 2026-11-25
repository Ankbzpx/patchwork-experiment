import argparse
from glob import glob
from itertools import combinations
import os

from fit_geo_3d import normalize_aabb, Patchwork
from misc import Timer

import igl
from joblib import delayed, Parallel
import numpy as np
import patchwork_query_ext
import scipy
import scipy.spatial
import torch

from icecream import ic
import polyscope as ps


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


def rm_duplicate_verts(V, F, tol=1e-5):

    duplicated_pairs = scipy.spatial.KDTree(V).query_pairs(
        tol, 2, output_type="ndarray"
    )
    NV = len(V)
    A = scipy.sparse.coo_matrix(
        (
            np.ones(len(duplicated_pairs)),
            (duplicated_pairs[:, 0], duplicated_pairs[:, 1]),
        ),
        shape=(NV, NV),
    )
    _, vid_remap = scipy.sparse.csgraph.connected_components(A, directed=False)
    _, unique_idx = np.unique(vid_remap, return_index=True)
    return V[unique_idx], vid_remap[F]


def region_convex_hull(A_equality, B_adj_indices, split):
    A_norm = np.linalg.norm(A_equality[:, :-1], axis=-1)
    c = np.zeros((A_equality.shape[1],))
    c[-1] = -1

    A = np.hstack((A_equality[:, :-1], A_norm[:, None]))
    b = -A_equality[:, -1:]

    res = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    if res.x is None or res.x[-1] < 0:
        return None, None, None, None

    interior_pt = res.x[:-1]

    # Guard possible numerical issue
    try:
        hs = scipy.spatial.HalfspaceIntersection(A_equality, interior_pt)
        hull = scipy.spatial.ConvexHull(hs.intersections)
    except:
        return None, None, None, None

    V = hs.intersections
    F = hull.simplices

    # Enforce consistently oriented normal
    v0 = V[F[:, 1]] - V[F[:, 0]]
    v1 = V[F[:, 2]] - V[F[:, 0]]
    FN = normalize(np.cross(v0, v1))
    V_bary = V[F].mean(1)

    dir = normalize(V_bary - interior_pt[None, :])
    dp = (dir * FN).sum(1)

    F = np.where(dp[:, None] > 0, F, F[:, ::-1])

    dists = np.abs(
        np.einsum(
            "nd,md->nm",
            A_equality,
            np.hstack([V_bary, np.ones((len(V_bary), 1))]),
        )
    )
    labels = np.argmin(dists, axis=0)
    labels = B_adj_indices[labels]
    boundary_mask = np.logical_and(dists.min(0) < 1e-10, labels < split)

    if boundary_mask.sum() == 0:
        return None, None, None, None

    labels = labels[boundary_mask]
    F = F[boundary_mask]
    unique_labels = np.unique(labels)

    V_sub_list = []
    F_sub_list = []
    base = len(V)

    cluster_area_list = []
    cluster_labels = []
    for label in unique_labels:
        loop = igl.boundary_loop(F[labels == label])
        V_loop = V[loop]

        F_sub = np.stack(
            [
                np.full(len(loop), base, dtype=np.int64),
                loop,
                np.roll(loop, -1),
            ],
            axis=-1,
        )
        base += 1
        V_sub_list.append(V_loop.mean(0))
        F_sub_list.append(F_sub)

        cluster_area = igl.doublearea(V, F[labels == label]).sum()
        cluster_area_list.append(np.full(len(F_sub), cluster_area))
        cluster_labels.append(np.full(len(F_sub), label))

    cluster_area = np.concatenate(cluster_area_list)
    cluster_labels = np.concatenate(cluster_labels)
    V_sub = np.vstack(V_sub_list)
    F_sub = np.vstack(F_sub_list)

    V_sub = np.vstack([V, V_sub])
    F_mask = cluster_area / cluster_area.max() < 0.1
    return V_sub, F_sub, F_mask, cluster_labels


def main(model_name: str, exp_dir: str, N: int, vis: bool, z: float, scale: float):
    tag = f"{model_name}"

    # Matches fit_geo_3d.py
    scale = min(scale, 2.0)

    if vis:
        gt_path = glob(os.path.join("data/mesh", f"{model_name}.*"))[0]
        V_gt, F_gt = igl.read_triangle_mesh(gt_path)
        V_gt = normalize_aabb(V_gt)

    model = Patchwork(np.empty((N, 3)), np.empty((N, 3)))
    save_path = os.path.join(exp_dir, tag, "model.pth")
    model.load_state_dict(torch.load(save_path))

    A, _, ga, ra2, B, _, gb, rb2 = model.unpack_compressed_coeffs()

    print(tag, len(B))

    A = A.detach().cpu().numpy()
    ga = ga.detach().cpu().numpy()
    ra2 = ra2.detach().cpu().numpy()
    B = B.detach().cpu().numpy()
    gb = gb.detach().cpu().numpy()
    rb2 = rb2.detach().cpu().numpy()

    g = np.vstack([ga, gb])
    r2 = np.concat([ra2, rb2])
    g_lift = np.hstack([g, (np.einsum("nd,nd->n", g, g) - r2)[:, None]])

    hull = scipy.spatial.ConvexHull(g_lift)

    pair_idx = np.array(list(combinations([0, 1, 2, 3], 2)))
    pairs = hull.simplices[:, pair_idx].reshape(-1, 2)

    Adj = scipy.sparse.coo_matrix(
        (
            np.ones((len(pairs) * 2,)),
            (
                np.concat([pairs[:, 0], pairs[:, 1]]),
                np.concat([pairs[:, 1], pairs[:, 0]]),
            ),
        ),
        (len(g), len(g)),
    ).tocsr()

    coeffs = np.vstack([A, B])

    def extract_hull(idx):
        B_adj_indices = Adj.indices[Adj.indptr[idx] : Adj.indptr[idx + 1]]
        A_equality = coeffs[B_adj_indices] - coeffs[idx][None, :]
        return region_convex_hull(A_equality, B_adj_indices, len(A))

    timer = Timer()
    timer.begin()
    rs = Parallel(n_jobs=16)(delayed(extract_hull)(i + len(A)) for i in range(len(B)))
    timer.end_data()

    # Detect unbounded
    bounds = []
    for V, F, C_mask, _ in rs:
        if V is not None and C_mask.sum() > 0:
            bounds.append(np.linalg.norm(V.max(0) - V.min(0)))
    bounds = np.array(bounds)

    # Upper fence: https://en.wikipedia.org/wiki/Interquartile_range
    log_bounds = np.log(bounds)
    q1, q3 = np.quantile(log_bounds, [0.25, 0.75])
    iqr = q3 - q1
    thr = np.exp(q3 + z * iqr)

    V_list = []
    F_list = []
    C_mask_list = []
    F_label_list = []
    F_label_list2 = []
    base = 0
    offset = 0
    for V, F, C_mask, F_label in rs:
        if V is not None:
            if np.linalg.norm(V.max(0) - V.min(0)) < thr:
                V_list.append(V)
                F_list.append(offset + F)
                C_mask_list.append(C_mask)
                F_label_list.append(F_label)
                F_label_list2.append(base * np.ones_like(F_label))
                offset += len(V)
                base += 1

    V = np.vstack(V_list)
    F = np.vstack(F_list)
    C_mask = np.concatenate(C_mask_list)
    F_label = np.concatenate(F_label_list)
    F_label2 = np.concatenate(F_label_list2)
    F_label[~C_mask] = -1
    F_label2[~C_mask] = -1

    # Extract the outer layers
    V, F = rm_duplicate_verts(V, F)
    V = V / scale

    F_mask = igl.doublearea(V, F) > 0
    F_label = F_label[F_mask]
    F_label2 = F_label2[F_mask]
    F = F[F_mask]

    V_clean, F_clean, _, _ = patchwork_query_ext.edge_collapse(V, F, F_label, F_label2)

    timer.end_iter()

    print(
        f"extract time: {timer.data_time}, combine time: {timer.iter_time}, total time: {timer.total_time()[-1]}\n"
    )

    if vis:
        ps.init()
        ms_viz = ps.register_surface_mesh("Mesh", V, F)
        ms_viz.add_scalar_quantity("F_label", F_label, defined_on="faces", enabled=True)
        ms_viz.add_scalar_quantity(
            "F_label2", F_label2, defined_on="faces", enabled=True
        )
        ps.register_surface_mesh("Mesh clean", V_clean, F_clean)
        ps.register_surface_mesh("GT", V_gt, F_gt)
        ps.show()
        exit()

    save_path = os.path.join(exp_dir, "result_meshes_explicit", f"{tag}.obj")
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    igl.write_triangle_mesh(save_path, V, F)

    save_path = os.path.join(exp_dir, "result_meshes_explicit_clean", f"{tag}.obj")
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    igl.write_triangle_mesh(save_path, V_clean, F_clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--exp_dir", type=str, help="Exp folder")
    parser.add_argument("--N", type=int, default=16384, help="Input point cloud size")
    parser.add_argument("--vis", action="store_true", help="Visualize")
    parser.add_argument("--z", type=float, default=3, help="z-score multiplier")
    parser.add_argument("--scale", type=float, default=1.0, help="Model scale")
    args = parser.parse_args()

    main(**vars(args))
