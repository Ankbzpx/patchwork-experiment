import os


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

from glob import glob
import json
import math
import shutil
import time

import equinox as eqx
import igl
import jax
from jax import grad, jit, numpy as jnp, vmap
from jaxtyping import Array, PyTree
from joblib import Memory
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


location = "__pycache__"
memory = Memory(location)


class Timer:
    def __init__(self):
        self.reset()

    def log(self, msg):
        cur_time = time.time()
        print(f"{msg}: {cur_time - self.start_time}")
        self.start_time = cur_time

    def reset(self):
        self.start_time = time.time()


def aabb_compute(V, scale=0.9):
    V_aabb_max = V.max(0, keepdims=True)
    V_aabb_min = V.min(0, keepdims=True)
    V_center = 0.5 * (V_aabb_max + V_aabb_min)
    scale = (V_aabb_max - V_center).max() / scale
    return V_center, scale, (V_aabb_max - V_aabb_min)


def normalize_aabb(V, scale=0.9):
    V_center, scale, _ = aabb_compute(V, scale)
    return (V - V_center) / scale


# Remove unreference vertices and assign new vertex indices
def rm_unref_vertices(V, F):
    V_unique, V_unique_idx, V_unique_idx_inv = np.unique(
        F.flatten(), return_index=True, return_inverse=True
    )
    V_id_new = np.arange(len(V_unique))
    V_map = V_id_new[np.argsort(V_unique_idx)]
    V_map_inv = np.zeros((np.max(V_map) + 1,), dtype=np.int64)
    V_map_inv[V_map] = V_id_new

    F = V_map_inv[V_unique_idx_inv].reshape(F.shape)
    V = V[V_unique][V_map]

    return V, F


@jit
def rot2d(theta):
    return jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )


@jit
def rot3d_x(theta):
    return jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, jnp.cos(theta), -jnp.sin(theta)],
            [0.0, jnp.sin(theta), jnp.cos(theta)],
        ]
    )


@jit
def rot3d_y(theta):
    return jnp.array(
        [
            [jnp.cos(theta), 0.0, jnp.sin(theta)],
            [0.0, 1.0, 0.0],
            [-jnp.sin(theta), 0.0, jnp.cos(theta)],
        ]
    )


@jit
def rot3d_z(theta):
    return jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta), 0.0],
            [jnp.sin(theta), jnp.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
@jit
def R3_from_vec_to_vec(a, b):
    a = normalize(a)
    b = normalize(b)

    cos = jnp.dot(a, b)
    sin = jnp.linalg.norm(jnp.cross(a, b))

    G = jnp.array(
        [
            [cos, -sin, 0],
            [sin, cos, 0],
            [0.0, 0.0, 1.0],
        ]
    )
    Fi = jnp.stack(
        [
            a,
            normalize(b - cos * a),
            normalize(jnp.cross(b, a)),
        ]
    )
    return Fi @ G @ jnp.linalg.inv(Fi)


@jit
def normalize_coeff(coeff):
    scale = jnp.sqrt(coeff[0] * coeff[0] + coeff[1] * coeff[1])
    return coeff / scale


def in_bound(x):
    return jnp.logical_and(x > 0, x < 1)


@jit
def line_to_end_points(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]

    coords = jnp.array([(a - c) / b, (b - c) / a, -(c + a) / b, -(b + c) / a])
    checks = vmap(in_bound)(coords)
    idx_sort = jnp.argsort(checks, descending=True)
    return jnp.array(
        [[-1, (a - c) / b], [(b - c) / a, -1], [1, -(c + a) / b], [-(b + c) / a, 1]]
    )[idx_sort][:2]


# Modified from: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    xyz = np.array(points)
    return xyz


@memory.cache
def mesh_sample_indicator(mesh_path, res):
    V, F = igl.read_triangle_mesh(mesh_path)
    V = normalize_aabb(V)
    line = np.linspace(-1.0, 1.0, res)
    samples = np.stack(np.meshgrid(line, line, line), -1).reshape(-1, 3)
    sample_sdfs = igl.signed_distance(samples, V, F)[0]
    return samples, np.where(sample_sdfs < 0, -1, 1)


def triangulate_height_map(height_map, res):
    xx, yy = np.meshgrid(np.arange(res), np.arange(res))
    idx = xx + yy * res
    N = res * res

    X = (xx / res).reshape(
        N,
    )
    Y = (yy / res).reshape(
        N,
    )
    vertices = np.stack((X, Y, height_map), -1)

    idx_valid = np.reshape(idx, (N))
    idx_map = np.zeros((N), dtype=np.int64)
    idx_map[idx_valid] = np.arange(idx_valid.shape[0])
    idx_map = idx_map.reshape((res, res))
    faces = []
    for r in range(res - 1):
        for c in range(res - 1):
            id00 = idx_map[r, c]
            id01 = idx_map[r, c + 1]
            id10 = idx_map[r + 1, c]
            id11 = idx_map[r + 1, c + 1]
            if id00 > 0 and id01 > 0 and id10 > 0 and id11 > 0:
                faces.append(np.array([id00, id11, id01]))
                faces.append(np.array([id00, id10, id11]))
            elif id00 > 0 and id01 > 0 and id11 > 0:
                faces.append(np.array([id00, id11, id01]))
            elif id00 > 0 and id10 > 0 and id11 > 0:
                faces.append(np.array([id00, id10, id11]))
            elif id00 > 0 and id01 > 0 and id10 > 0:
                faces.append(np.array([id00, id10, id01]))
            elif id01 > 0 and id10 > 0 and id11 > 0:
                faces.append(np.array([id01, id10, id11]))
    faces = np.stack(faces, axis=0)
    return vertices, faces


def batch_call(
    func, input, num_out_args=1, out_map_func=lambda x: [x], group_size=128**2
):
    n_iters = len(input) // group_size

    if n_iters == 0:
        output = func(input)
        output = out_map_func(output)
    else:
        output = {}
        for i in range(num_out_args):
            output[i] = None

        input_splits = jnp.array_split(input, n_iters)
        for input_batch in input_splits:
            output_ = func(input_batch)
            output_ = out_map_func(output_)

            for i in range(num_out_args):
                output[i] = (
                    output_[i]
                    if output[i] is None
                    else jnp.concatenate([output[i], output_[i]])
                )

        output = list(output.values())

    if num_out_args == 1:
        output = output[0]

    return output


# infer: R^3 -> R
def voxel_infer(
    infer,
    grid_res=512,
    grid_bl=np.array([-1.0, -1.0, -1.0]),
    grid_tr=np.array([1.0, 1.0, 1.0]),
    group_size=16384,
    out_dim=1,
):
    iter_size = grid_res**3 // group_size

    # Cannot pass jitted function as argument to another jitted function
    @jit
    def infer_scalar():
        # For consistency with partition, we ignore the endpoint
        idx_x = jnp.linspace(grid_bl[0], grid_tr[0], grid_res, endpoint=False)
        idx_y = jnp.linspace(grid_bl[1], grid_tr[1], grid_res, endpoint=False)
        idx_z = jnp.linspace(grid_bl[2], grid_tr[2], grid_res, endpoint=False)
        grid = jnp.stack(jnp.meshgrid(idx_x, idx_y, idx_z), -1)

        query_data = {
            "grid": grid.reshape(iter_size, group_size, 3),
            "val": jnp.zeros((iter_size, group_size, out_dim)),
        }

        @jit
        def body_func(i, query_data):
            val = infer(query_data["grid"][i]).reshape(-1, out_dim)
            query_data["val"] = query_data["val"].at[i].set(val)
            return query_data

        query_data = jax.lax.fori_loop(0, iter_size, body_func, query_data)
        return query_data["val"].reshape(grid_res, grid_res, grid_res, out_dim), grid

    return infer_scalar()


# infer: R^3 -> R (sdf)
def extract_surface(
    infer, grid_res=512, group_size=16384, grid_min=-1.0, grid_max=1.0, iso=0.0
):
    grid_max_res = 512

    if grid_res > grid_max_res:
        # Have to partition
        div = int(np.ceil(grid_res / grid_max_res))
        interval = (grid_max - grid_min) / div

        part_idx = np.arange(div)
        part_offsets = np.stack(np.meshgrid(part_idx, part_idx, part_idx), -1).reshape(
            -1, 3
        )

        part_bl = grid_min + part_offsets * interval
        part_tr = grid_min + (part_offsets + 1) * interval

        block_list = []
        for i in range(div**3):
            sdf, _ = voxel_infer(
                infer, grid_max_res, grid_bl=part_bl[i], grid_tr=part_tr[i]
            )
            block_list.append(np.array(sdf[..., 0]))

        sdf_np = np.stack(block_list).reshape(
            div, div, div, grid_max_res, grid_max_res, grid_max_res
        )
        sdf_np = np.transpose(sdf_np, (0, 3, 1, 4, 2, 5)).reshape(
            grid_res, grid_res, grid_res
        )
    else:
        sdf, _ = voxel_infer(
            infer,
            grid_res,
            grid_bl=np.array([grid_min, grid_min, grid_min]),
            grid_tr=np.array([grid_max, grid_max, grid_max]),
            group_size=group_size,
        )
        # This step is surprising slow, gpu to cpu memory copy?
        sdf_np = np.array(sdf[..., 0])

    sdf_np = np.swapaxes(sdf_np, 0, 1)
    spacing = 1.0 / grid_res
    # It outputs inverse VN, even with gradient_direction set to ascent
    V, F, VN_inv, _ = marching_cubes(sdf_np, iso, spacing=(spacing, spacing, spacing))
    dim = grid_max - grid_min
    V = dim * (V - np.abs(grid_min) / dim)
    return V, F, -VN_inv


class CheckPointer:
    def __init__(self, exp_dir="exp"):
        self.exp_dir = exp_dir

    def backup(self, save_dir):
        backup_dir = os.path.join(save_dir, "backup")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        file_list = glob("**/*.py", recursive=True)
        for file_path in file_list:
            file_save_path = os.path.join(backup_dir, file_path)

            # Avoid recursive save
            if file_path.split("/")[0].startswith(self.exp_dir):
                continue

            # Ignore cache folder
            if file_path.split("/")[0].startswith("__pycache__"):
                continue

            file_save_base = os.path.dirname(file_save_path)
            if not os.path.exists(file_save_base):
                os.makedirs(file_save_base)

            shutil.copy(file_path, file_save_path)

    def serialize(self, model: PyTree, tag, meta=None):
        save_dir = os.path.join(self.exp_dir, tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # self.backup(save_dir)
        eqx.tree_serialise_leaves(os.path.join(save_dir, "model.eqx"), model)

        if meta is not None:
            with open(os.path.join(save_dir, "meta.json"), "w") as f:
                json.dump(meta, f)

    def deserialize(self, model: PyTree, tag, meta_callback=None) -> PyTree:
        save_dir = os.path.join(self.exp_dir, tag)
        if meta_callback is not None:
            with open(os.path.join(save_dir, "meta.json"), "r") as f:
                meta = json.load(f)
                model = meta_callback(model, meta)
        return eqx.tree_deserialise_leaves(os.path.join(save_dir, "model.eqx"), model)


# Reference: https://github.com/Chumbyte/DiGS/blob/main/surface_reconstruction/compute_metrics_srb.py
def compute_metrics(recon_points, gt_points, f1_thr, n_worker=8):
    recon_kd_tree = cKDTree(recon_points)
    gt_kd_tree = cKDTree(gt_points)
    re2gt_distances, _ = recon_kd_tree.query(gt_points, workers=n_worker)
    gt2re_distances, _ = gt_kd_tree.query(recon_points, workers=n_worker)
    cd_re2gt = np.mean(re2gt_distances)
    cd_gt2re = np.mean(gt2re_distances)
    hd_re2gt = np.max(re2gt_distances)
    hd_gt2re = np.max(gt2re_distances)
    chamfer_dist = 0.5 * (cd_re2gt + cd_gt2re)
    hausdorff_distance = np.max((hd_re2gt, hd_gt2re))

    precision = np.sum(re2gt_distances < f1_thr) / len(gt_points)
    recall = np.sum(gt2re_distances < f1_thr) / len(recon_points)
    f_1_score = 2 * precision * recall / (precision + recall + 1e-8)

    return chamfer_dist, hausdorff_distance, f_1_score
