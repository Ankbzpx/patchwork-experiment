from glob import glob
import os

from common import normalize_aabb

import igl
from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import trimesh
import warp as wp

from icecream import ic
import polyscope as ps


@wp.kernel
def closest_query_kernel(
    in_x: wp.array(dtype=wp.vec3),
    geom: wp.uint64,
    out_sign: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    x = in_x[i]

    query = wp.mesh_query_point_sign_winding_number(geom, x, 2.0)
    if query.result:
        out_sign[i] = query.sign


# Reference: https://github.com/Chumbyte/DiGS/blob/main/surface_reconstruction/compute_metrics_srb.py
def compute_metrics(
    recon_points, gt_points, recon_normals, gt_normals, f1_thr, n_worker=8
):
    recon_kd_tree = cKDTree(recon_points)
    gt_kd_tree = cKDTree(gt_points)
    re2gt_distances, _ = recon_kd_tree.query(gt_points, workers=n_worker)
    gt2re_distances, gt2re_indices = gt_kd_tree.query(recon_points, workers=n_worker)

    # https://github.com/autonomousvision/convolutional_occupancy_networks/blob/838bea5b2f1314f2edbb68d05ebb0db49f1f3bd2/src/eval.py#L189
    normals_acc = (gt_normals[gt2re_indices] * recon_normals).sum(axis=-1)
    normals_acc = np.abs(normals_acc).mean()

    cd_re2gt = np.mean(re2gt_distances)
    cd_gt2re = np.mean(gt2re_distances)
    hd_re2gt = np.max(re2gt_distances)
    hd_gt2re = np.max(gt2re_distances)
    chamfer_dist = 0.5 * (cd_re2gt + cd_gt2re)
    hausdorff_distance = np.max((hd_re2gt, hd_gt2re))

    precision = np.sum(re2gt_distances < f1_thr) / len(gt_points)
    recall = np.sum(gt2re_distances < f1_thr) / len(recon_points)
    f_1_score = 2 * precision * recall / (precision + recall + 1e-8)

    return chamfer_dist, hausdorff_distance, f_1_score, normals_acc


def process_method(method, dataset):
    wp.init()

    def query_sign(x, V, F):
        geom_mesh = wp.Mesh(
            points=wp.array(V, dtype=wp.vec3),
            indices=wp.array(
                F.reshape(
                    -1,
                ),
                dtype=int,
            ),
        )

        N = len(x)
        x_wp = wp.array(x, dtype=wp.vec3)
        signs_wp = wp.empty(shape=(N,), dtype=wp.float32)
        wp.launch(
            closest_query_kernel,
            dim=N,
            inputs=(x_wp, geom_mesh.id),
            outputs=(signs_wp,),
        )

        return wp.to_torch(signs_wp).detach().cpu().numpy() < 0

    missing_cases = []
    initialized_tags = set()

    collection = {}
    gt_folder = os.path.join(gt_root, dataset)
    model_list = glob(
        os.path.join(gt_folder, "**" if dataset == "shapenet" else "", "*.*")
    )
    model_list = sorted([model.replace(gt_folder, "")[1:] for model in model_list])

    for model in tqdm(model_list):
        V, F = igl.read_triangle_mesh(os.path.join(gt_folder, model))
        V = normalize_aabb(V)
        gt_mesh: trimesh.Trimesh = trimesh.Trimesh(V, F)
        gt_samples, gt_fid = trimesh.sample.sample_surface(
            gt_mesh, sample_size, seed=seed
        )
        gt_normals = gt_mesh.face_normals[gt_fid]
        aabb = gt_mesh.bounding_box.bounds
        max_bound = np.max(aabb[1] - aabb[0])
        f1_thr = f1_percent * max_bound

        # https://github.com/autonomousvision/convolutional_occupancy_networks/blob/838bea5b2f1314f2edbb68d05ebb0db49f1f3bd2/src/common.py#L30
        volume_samples = np.random.uniform(-1.0, 1.0, (sample_size, 3))
        gt_occ = query_sign(volume_samples, gt_mesh.vertices, gt_mesh.faces)

        def append_collection(tag, result_samples, result_normals, iou):
            chamfer_dist, hausdorff_distance, f_1_score, normal_acc = compute_metrics(
                result_samples, gt_samples, result_normals, gt_normals, f1_thr
            )

            metrics_frame = pd.DataFrame(
                [
                    [
                        model_name,
                        chamfer_dist,
                        hausdorff_distance,
                        f_1_score,
                        normal_acc,
                        iou,
                    ]
                ],
                columns=metrics_column,
            )

            if tag not in collection:
                collection[tag] = metrics_frame
            else:
                collection[tag] = pd.concat([collection[tag], metrics_frame])

            csv_path = os.path.join("metrics", f"{tag}.csv")
            if tag not in initialized_tags:
                metrics_frame.to_csv(csv_path, mode="w", header=True, index=False)
                initialized_tags.add(tag)
            else:
                metrics_frame.to_csv(csv_path, mode="a", header=False, index=False)

        model_name = model.split(".")[0]
        tag = f"{method}_{dataset}"
        result_path = glob(
            os.path.join(
                result_root,
                method,
                f"{dataset}",
                f"{model_name}*",
            )
        )

        if len(result_path) < 1:
            continue

        result_path = result_path[0]

        V, F = igl.read_triangle_mesh(result_path)

        if len(V) == 0:
            # FIXME: Skip for now
            missing_cases.append((method, model_name))
        else:
            result_mesh: trimesh.Trimesh = trimesh.Trimesh(V, F)
            result_samples, result_fid = trimesh.sample.sample_surface(
                result_mesh, sample_size, seed=seed
            )
            result_normals = result_mesh.face_normals[result_fid]

            result_occ = query_sign(
                volume_samples, result_mesh.vertices, result_mesh.faces
            )
            area_union = (gt_occ | result_occ).astype(np.float32).sum(axis=-1)
            area_intersect = (gt_occ & result_occ).astype(np.float32).sum(axis=-1)
            iou = area_intersect / (area_union + 1e-8)

            append_collection(tag, result_samples, result_normals, iou)

    return missing_cases


if __name__ == "__main__":
    gt_root = os.path.expandvars("data/mesh")
    result_root = os.path.expandvars("./")

    dataset_list = ["abc", "thingi10k", "roof_modeling", "shapenet"]
    method_list = [
        "ours_explicit",
    ]

    seed = 0
    sample_size = 1000000
    f1_percent = 1e-3

    np.random.seed(seed)

    os.makedirs("metrics", exist_ok=True)

    config_list = []
    for method in method_list:
        for dataset in dataset_list:
            if os.path.exists(os.path.join(result_root, method, dataset)):
                config_list.append((method, dataset))

    metrics_column = ["item", "chamfer", "hausdorff", "f1", "normal", "iou"]

    results = Parallel(
        n_jobs=min(len(config_list), os.cpu_count() // 2),
        backend="multiprocessing",
        prefer="processes",  # use separate processes (better for CPU-heavy tasks)
        verbose=10,
    )(delayed(process_method)(method, dataset) for (method, dataset) in config_list)

    print(results)
