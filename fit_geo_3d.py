import argparse
from glob import glob
import json
import os
import random

from common import normalize_aabb
from triton_kernel import logsumexp_s

import igl
from misc import Timer
import numpy as np
from patchwork_query_ext.kdtree import KDTree
from skimage.measure import marching_cubes
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, get_worker_info, IterableDataset
from tqdm import tqdm
import trimesh

from icecream import ic
import polyscope as ps


class Patchwork(nn.Module):
    def __init__(self, pc: Tensor, pc_normal: Tensor, offset=1e-2):
        super().__init__()

        center = torch.from_numpy(pc).float()
        dir = F.normalize(torch.from_numpy(pc_normal).float())
        N = len(center)

        self.ga = nn.Parameter(center + offset * dir)
        self.gb = nn.Parameter(center - offset * dir)

        self.ra2 = nn.Parameter(torch.zeros((N,)))
        self.rb2 = nn.Parameter(torch.zeros((N,)))

        self.s1 = nn.Parameter(2e-2 * torch.ones((N,)))
        self.s2 = nn.Parameter(2e-2 * torch.ones((N,)))

    def unpack_coeffs(self) -> list[Tensor]:
        a = 2 * self.ga
        b = 2 * self.gb

        ra2 = F.softplus(1e-1 * self.ra2)
        rb2 = F.softplus(1e-1 * self.rb2)

        c = -torch.square(self.ga).sum(1) + ra2
        d = -torch.square(self.gb).sum(1) + rb2

        s1 = F.sigmoid(1e2 * self.s1)
        s2 = F.sigmoid(1e2 * self.s2)

        A = torch.hstack([a, c[:, None]])
        B = torch.hstack([b, d[:, None]])

        return A, s1, self.ga, ra2, B, s2, self.gb, rb2

    def unpack_compressed_coeffs(self) -> list[Tensor]:
        A, s1, ga, ra2, B, s2, gb, rb2 = self.unpack_coeffs()
        s1_mask = s1 > 0.5
        s2_mask = s2 > 0.5
        return (
            A[s1_mask],
            torch.ones_like(s1[s1_mask]),
            ga[s1_mask],
            ra2[s1_mask],
            B[s2_mask],
            torch.ones_like(s2[s2_mask]),
            gb[s2_mask],
            rb2[s2_mask],
        )


class SurfaceSampleDataset(IterableDataset):
    def __init__(
        self, mesh_tri: trimesh.Trimesh, sample_size: int = 16384, seed: int = 0
    ):
        super().__init__()
        self.mesh_tri = mesh_tri
        self.sample_size = sample_size
        self.seed = seed

    def sample(self, rng):
        sur_samples, fids = trimesh.sample.sample_surface(
            self.mesh_tri, self.sample_size, seed=rng
        )
        sur_normals = self.mesh_tri.face_normals[fids]

        sur_samples = np.asarray(sur_samples).astype(np.float32)
        sur_normals = np.asarray(sur_normals).astype(np.float32)

        return {"sur_samples": sur_samples, "sur_normals": sur_normals}

    def __iter__(self):
        rng = np.random.default_rng(get_worker_info().seed)
        while True:
            yield self.sample(rng)


# Symmetry preserves the boundary, so it forces moving the generators to change the assignment
def symmetric_reparam(a: Tensor, b: Tensor) -> Tensor:
    return (0.5 * (a + b) + 0.5 * (a - b).detach()) - (
        0.5 * (a + b) - 0.5 * (a - b).detach()
    )


# Anti-symmetry changes the difference of power to shift the level set
def anti_symmetric_reparam(a: Tensor, b: Tensor) -> Tensor:
    return (0.5 * (a + b).detach() + 0.5 * (a - b)) - (
        0.5 * (a + b).detach() - 0.5 * (a - b)
    )


def eval_mse_res(X: Tensor, ga: Tensor, gb: Tensor, ra2: Tensor, rb2: Tensor) -> Tensor:
    return torch.abs(
        torch.einsum(
            "nd,nd->n",
            (ga + gb) - 2 * X,
            (ga - gb).detach(),
        )
        - anti_symmetric_reparam(ra2, rb2)
    )


def eval_normal_res(XN: Tensor, ga: Tensor, gb: Tensor) -> Tensor:
    return F.cosine_similarity(
        ga - gb,
        XN,
    )


def eval_cover_res(X, ga, ga_sec, ra2, ra2_sec) -> Tensor:
    return F.relu(
        torch.einsum(
            "nd,nd->n",
            (ga_sec + ga) - 2 * X,
            (ga_sec - ga),
        )
        - anti_symmetric_reparam(ra2_sec, ra2)
    )


def eval_power_dist_flipped(X, ga, ra2) -> Tensor:
    return (
        torch.einsum(
            "nd,nd->n",
            2 * ga,
            X,
        )
        - torch.square(ga).sum(1)
        + ra2
    )


def extract_mc(
    func: callable,
    grid_res=256,
    low=np.array([-1.0, -1.0, -1.0]),
    high=np.array([1.0, 1.0, 1.0]),
    group_size=65536,
):
    grid_pts = torch.stack(
        torch.meshgrid(
            torch.linspace(low[0], high[0], grid_res),
            torch.linspace(low[1], high[1], grid_res),
            torch.linspace(low[2], high[2], grid_res),
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 3)

    sdfs = []
    for pt_group in tqdm(torch.split(grid_pts, group_size)):
        sdf_group = func(pt_group.float().cuda()).detach().cpu()
        sdfs.append(sdf_group)
    sdfs = torch.concat(sdfs)
    sdfs = sdfs.reshape(grid_res, grid_res, grid_res).numpy()

    spacing = 1.0 / (grid_res - 1)
    V, F, _, _ = marching_cubes(sdfs, 0, spacing=(spacing, spacing, spacing))
    V = low[None, :] + (high[None, :] - low[None, :]) * V
    return V, F


def main(
    model_name: str,
    exp_dir: str,
    N: int,
    batch_size: int,
    offset: float,
    w_mse: float,
    w_normal: float,
    w_cover: float,
    w_prune: float,
    lr: float,
    n_iters: int,
    res: int,
    scale: float,
    theta: float,
    eval: bool,
    beta: float,
    vis: bool,
):
    np.random.seed(0)
    torch.manual_seed(0)

    tag = f"{model_name}"

    if eval:
        model = Patchwork(np.empty((N, 3)), np.empty((N, 3)))
        save_path = os.path.join(exp_dir, tag, "model.pth")
        model.load_state_dict(torch.load(save_path))
        model = model.cuda()

        A, s1, ga, ra2, B, s2, gb, rb2 = model.unpack_compressed_coeffs()

        if beta == 0:
            save_name = f"{tag}.obj"
            tree_a = KDTree(ga, ra2, s1)
            tree_b = KDTree(gb, rb2, s2)

            def _eval(X: Tensor):
                h_idx = tree_a.query_closest(X)
                g_idx = tree_b.query_closest(X)

                sdf = eval_power_dist_flipped(
                    X, ga[h_idx], ra2[h_idx]
                ) - eval_power_dist_flipped(X, gb[g_idx], rb2[g_idx])
                return sdf
        else:
            save_name = f"{tag}_{beta}.obj"
            beta_1 = beta * s1.new_ones((1,))
            beta_2 = beta * s2.new_ones((1,))

            def _eval(X: Tensor):
                X_pad = F.pad(X, (0, 1), "constant", 1)
                h, _, _, _, _, _, _ = logsumexp_s(X_pad, A, s1, beta_1)
                g, _, _, _, _, _, _ = logsumexp_s(X_pad, B, s2, beta_2)
                return h - g

        V_mc, F_mc = extract_mc(
            _eval,
            grid_res=res,
            low=1.1 * ga.min(0)[0].detach().cpu().numpy(),
            high=1.1 * ga.max(0)[0].detach().cpu().numpy(),
        )

        save_path = os.path.join(exp_dir, "result_meshes", save_name)
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        igl.write_triangle_mesh(save_path, V_mc, F_mc)
        exit()

    # When scale >= 2.0, we scale model internally instead,
    #   due to limited FP32 precision when computing the maximum
    tau = 1.0
    if scale <= 2.0:
        tau = scale
        scale = 1.0

    gt_path = glob(os.path.join("data/mesh", f"{model_name}.*"))[0]
    V_gt, F_gt = igl.read_triangle_mesh(gt_path)
    V_gt = normalize_aabb(V_gt, 0.9 * scale)

    mesh_tri = trimesh.Trimesh(V_gt, F_gt)
    sur_samples, fids = trimesh.sample.sample_surface(mesh_tri, N, seed=0)
    sur_normals = mesh_tri.face_normals[fids]

    sur_samples = np.asarray(sur_samples)
    sur_normals = np.asarray(sur_normals)

    model: Patchwork = Patchwork(sur_samples, sur_normals, offset / tau).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr / tau)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(0.4 * n_iters), 0.25)

    s1_mask = torch.ones_like(model.s1).bool()
    s2_mask = torch.ones_like(model.s2).bool()

    def train_step(
        model: Patchwork, iter: int, sur_samples: Tensor, sur_normals: Tensor
    ):
        nonlocal s1_mask, s2_mask

        sur_samples = sur_samples.float().cuda(non_blocking=True)
        sur_normals = sur_normals.float().cuda(non_blocking=True)

        A, s1, ga, ra2, B, s2, gb, rb2 = model.unpack_coeffs()

        # X = F.pad(sur_samples, (0, 1), "constant", 1)
        # _, _, _, _, h_idx, h_sec_idx, _ = logsumexp_s(
        #     X, A, torch.where(s1 > 0.5, 1.0, 1e-8), s1.new_ones((1,))
        # )
        # _, _, _, _, g_idx, g_sec_idx, _ = logsumexp_s(
        #     X, B, torch.where(s2 > 0.5, 1.0, 1e-8), s2.new_ones((1,))
        # )

        tree_a = KDTree(ga, ra2, torch.where(s1 > 0.5, 1.0, 1e-8))
        tree_b = KDTree(gb, rb2, torch.where(s2 > 0.5, 1.0, 1e-8))

        h_idx, h_sec_idx = tree_a.query_knn(sur_samples, 2)
        g_idx, g_sec_idx = tree_b.query_knn(sur_samples, 2)

        ra2_h = ra2[h_idx]
        rb2_g = rb2[g_idx]

        ra2_h_sec = ra2[h_sec_idx]
        rb2_g_sec = rb2[g_sec_idx]

        ga_h = ga[h_idx]
        gb_g = gb[g_idx]

        ga_h_sec = ga[h_sec_idx]
        gb_g_sec = gb[g_sec_idx]

        candidate_normals = torch.stack(
            [
                F.normalize(ga_h - gb_g),
                F.normalize(ga_h - gb_g_sec),
                F.normalize(ga_h_sec - gb_g),
                F.normalize(ga_h_sec - gb_g_sec),
            ],
            dim=1,
        )

        h_idx_candidate = torch.stack([h_idx, h_idx, h_sec_idx, h_sec_idx], dim=1)
        g_idx_candidate = torch.stack([g_idx, g_sec_idx, g_idx, g_sec_idx], dim=1)

        candidate_mse_res = torch.stack(
            [
                eval_mse_res(sur_samples, ga_h, gb_g, ra2_h, rb2_g),
                eval_mse_res(sur_samples, ga_h, gb_g_sec, ra2_h, rb2_g_sec),
                eval_mse_res(sur_samples, ga_h_sec, gb_g, ra2_h_sec, rb2_g),
                eval_mse_res(sur_samples, ga_h_sec, gb_g_sec, ra2_h_sec, rb2_g_sec),
            ],
            dim=1,
        )

        candidate_normal_res = torch.stack(
            [
                eval_normal_res(sur_normals, ga_h, gb_g),
                eval_normal_res(sur_normals, ga_h, gb_g_sec),
                eval_normal_res(sur_normals, ga_h_sec, gb_g),
                eval_normal_res(sur_normals, ga_h_sec, gb_g_sec),
            ],
            dim=1,
        )

        candidate_cover_res = torch.stack(
            [
                torch.zeros_like(ra2_h),
                eval_cover_res(sur_samples, gb_g, gb_g_sec, rb2_g, rb2_g_sec),
                eval_cover_res(sur_samples, ga_h, ga_h_sec, ra2_h, ra2_h_sec),
                eval_cover_res(sur_samples, ga_h, ga_h_sec, ra2_h, ra2_h_sec)
                + eval_cover_res(sur_samples, gb_g, gb_g_sec, rb2_g, rb2_g_sec),
            ],
            dim=1,
        )

        dp = torch.einsum("bnd,bd->bn", candidate_normals, sur_normals)

        # Hard assignment seems better than Softmax with small temperature
        pick = torch.argmax(dp, dim=1)
        dp_pick = torch.take_along_dim(dp, pick[:, None], dim=1)[:, 0]
        d_theta = torch.arccos(torch.clip(dp[:, 0], -1, 1)) - torch.arccos(
            torch.clip(dp_pick, -1, 1)
        )
        # Keep the current assignment if they are close enough
        pick = torch.where(torch.abs(d_theta) > np.deg2rad(theta), pick, 0)

        h_idx_pick = torch.take_along_dim(
            h_idx_candidate,
            pick[:, None],
            dim=1,
        )[:, 0]
        g_idx_pick = torch.take_along_dim(
            g_idx_candidate,
            pick[:, None],
            dim=1,
        )[:, 0]

        loss_mse = torch.take_along_dim(candidate_mse_res, pick[:, None], dim=1).mean()
        norm_res_pick = torch.take_along_dim(
            candidate_normal_res, pick[:, None], dim=1
        )[:, 0]
        loss_normal = (1 - torch.where(norm_res_pick > 0, norm_res_pick, 0.0)).mean()
        loss_cover = torch.take_along_dim(
            candidate_cover_res, pick[:, None], dim=1
        ).mean()

        loss = w_mse * loss_mse + w_normal * loss_normal + w_cover * loss_cover

        s1_mask[
            h_idx_candidate.reshape(
                -1,
            )
        ] = False
        s1_mask[h_idx_pick] = True

        s2_mask[
            g_idx_candidate.reshape(
                -1,
            )
        ] = False
        s2_mask[g_idx_pick[norm_res_pick > 0]] = True

        loss_prune = (
            torch.where(s1_mask, torch.abs(1 - s1), torch.abs(s1)).mean()
            + torch.where(s2_mask, torch.abs(1 - s2), torch.abs(s2)).mean()
        )
        loss += w_prune * loss_prune

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        lr_scheduler.step()

        return {
            "loss": loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_normal": loss_normal.item(),
            "loss_cover": loss_cover.item(),
            "loss_prune": loss_prune.item(),
        }

    dataset = SurfaceSampleDataset(mesh_tri, batch_size)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=8,
        worker_init_fn=seed_worker,
        pin_memory=True,
        generator=g,
    )
    dataloader_iter = iter(dataloader)

    timer = Timer()
    pbar = tqdm(range(n_iters), dynamic_ncols=True)
    for it in pbar:
        timer.begin()
        batch = next(dataloader_iter)
        timer.end_data()
        loss_dict = train_step(model, it, **batch)
        timer.end_iter()
        pbar.set_postfix(loss_dict)

        if it != 0 and it % 1000 == 0:
            print(
                f"data time: {timer.data_time}, iter time: {timer.iter_time}, total time: {timer.total_time()[-1]}\n"
            )

    elapsed = timer.total_time()[0]

    save_dir = os.path.join(exp_dir, tag)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), save_path)

    A, s1, ga, ra2, B, s2, gb, rb2 = model.unpack_compressed_coeffs()
    tree_a = KDTree(ga, ra2, s1)
    tree_b = KDTree(gb, rb2, s2)

    n_params = len(A) + len(B)
    meta = {"n_params": n_params, "elapsed": elapsed}
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    def _eval(X: Tensor):
        h_idx = tree_a.query_closest(X)
        g_idx = tree_b.query_closest(X)

        sdf = eval_power_dist_flipped(
            X, ga[h_idx], ra2[h_idx]
        ) - eval_power_dist_flipped(X, gb[g_idx], rb2[g_idx])
        return sdf

    V_mc, F_mc = extract_mc(
        _eval, grid_res=res, low=1.1 * V_gt.min(0), high=1.1 * V_gt.max(0)
    )
    V_mc /= scale
    save_path = os.path.join(exp_dir, "result_meshes", f"{tag}.obj")
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    igl.write_triangle_mesh(save_path, V_mc, F_mc)

    if vis:
        ps.init()
        ps.register_point_cloud("ga", ga.detach().cpu() / scale)
        ps.register_point_cloud("gb", gb.detach().cpu() / scale)
        ps.register_surface_mesh("GT", V_gt / scale, F_gt)
        ps.register_surface_mesh("MC", V_mc, F_mc)
        ps.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--exp_dir", type=str, help="Exp folder")
    parser.add_argument("--N", type=int, default=16384, help="Input point cloud size")
    parser.add_argument("--batch_size", type=int, default=16384, help="Batch size")
    parser.add_argument("--offset", type=float, default=1e-2, help="Init offset")
    parser.add_argument("--w_mse", type=float, default=10.0, help="Weight mse")
    parser.add_argument("--w_normal", type=float, default=1.0, help="Weight normal")
    parser.add_argument("--w_cover", type=float, default=100.0, help="Weight reg")
    parser.add_argument("--w_prune", type=float, default=1.0, help="Weight prune")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--n_iters", type=int, default=10000, help="Number of iterations"
    )
    parser.add_argument("--res", type=int, default=256, help="MC res")
    parser.add_argument("--scale", type=float, default=1.0, help="Model scale")
    parser.add_argument("--theta", type=float, default=5.0, help="Angular tolerance")
    parser.add_argument("--eval", action="store_true", help="Evaluate only")
    parser.add_argument("--beta", type=float, default=0, help="Beta")
    parser.add_argument("--vis", action="store_true", help="Visualize extraction")
    args = parser.parse_args()

    main(**vars(args))
