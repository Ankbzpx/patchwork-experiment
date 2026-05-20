import argparse
from glob import glob
import json
import os

from triton_kernel import logsumexp

import igl
from jaxtyping import Float
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    PreTrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from icecream import ic
import polyscope as ps


flex_flash = torch.compile(flex_attention, dynamic=False)


def aabb_compute(V, scale=0.9):
    V_aabb_max = V.max(0, keepdims=True)
    V_aabb_min = V.min(0, keepdims=True)
    V_center = 0.5 * (V_aabb_max + V_aabb_min)
    scale = (V_aabb_max - V_center).max() / scale
    return V_center, scale, (V_aabb_max - V_aabb_min)


def normalize_aabb(V, scale=0.9):
    V_center, scale, _ = aabb_compute(V, scale)
    return (V - V_center) / scale


def double_well_potential(x):
    x = 2 * (x - 0.5)
    return torch.square(x) - 2 * torch.abs(x) + 1


# The gradient of LSE
def grad_LSE(X, A, c, s, beta):
    # embedding dimension of the query, key, and value must be at least 16
    D = X.shape[-1]
    D_pad = max(0, 16 - D)

    Q = F.pad(X, (0, D_pad))[None, None, ...]
    K = F.pad(A, (0, D_pad))[None, None, ...]
    V = F.pad(A, (0, D_pad))[None, None, ...]

    log_s = torch.log(s)

    def score_mod(score, batch, head, _q_idx, k_idx):
        return beta[0] * (score + c[k_idx]) + log_s[k_idx]

    out = flex_flash(Q, K, V, score_mod=score_mod, scale=1.0)
    return out[0, 0, :, :D]


class SDFDataset(Dataset):
    def __init__(self, model_name: str, n_steps, n_samples):
        super().__init__()

        pc_path = f"data/pc/{model_name}.ply"
        pc_o3d = o3d.io.read_point_cloud(os.path.expandvars(pc_path))
        sur_samples = np.asarray(pc_o3d.points)[:n_samples]
        sur_normals = np.asarray(pc_o3d.normals)[:n_samples]

        sigma_set = []
        ptree = cKDTree(sur_samples)

        for p in np.array_split(sur_samples, 100, axis=0):
            d = ptree.query(p, 50 + 1)
            sigma_set.append(d[0][:, -1])

        self.global_sigma = 1.8
        self.local_sigma = torch.from_numpy(np.concatenate(sigma_set)).float()
        self.sur_samples = torch.from_numpy(sur_samples).float()
        self.sur_normals = torch.from_numpy(sur_normals).float()

        self.n_steps = n_steps
        self.n_samples = n_samples

    def __len__(self):
        return self.n_steps

    def __getitem__(self, index):

        sample_local = (
            self.sur_samples
            + torch.randn_like(self.sur_samples) * self.local_sigma[:, None]
        )

        return {
            "sur_samples": self.sur_samples,
            "sur_normals": self.sur_normals,
            "samples": sample_local,
        }


class PatchworkConfig(PreTrainedConfig):
    model_type = "patchwork"

    def __init__(
        self,
        rho: float = 200.0,
        beta: float = 75.0,
        w_mse: float = 1.0,
        w_normal: float = 1.0,
        w_reg: float = 1.0,
        w_prune: float = 1.0,
        grad_pass_through: bool = False,
    ):
        super().__init__()
        self.rho = rho
        self.beta = beta
        self.w_mse = w_mse
        self.w_normal = w_normal
        self.w_reg = w_reg
        self.w_prune = w_prune
        self.accurate_grad = not grad_pass_through


class Patchwork(PreTrainedModel):
    def __init__(
        self,
        config: PatchworkConfig,
        sur_samples: Float[Tensor, "n 3"],
        sur_normals: Float[Tensor, "n 3"],
    ):
        super().__init__(config)

        a = config.rho * sur_samples + sur_normals
        b = config.rho * sur_samples
        l2 = 0.5 * config.rho * np.einsum("ni,ni->n", sur_samples, sur_samples)
        c1 = l2 - np.einsum("ni,ni->n", a, sur_samples)
        c2 = l2 - np.einsum("ni,ni->n", b, sur_samples)

        a_norm = np.linalg.norm(a, axis=-1)
        a_dir = a / (a_norm[:, None] + 1e-8)
        s1 = np.log(np.e - 1) * np.ones_like(c1)

        self.a_dir = nn.Parameter(torch.from_numpy(a_dir).float())
        self.a_norm = nn.Parameter(torch.from_numpy(a_norm).float())
        self.c1 = nn.Parameter(torch.from_numpy(c1).float())
        self.s1 = nn.Parameter(torch.from_numpy(s1).float())
        self.beta1 = nn.Parameter(torch.Tensor([config.beta]).float())

        b_norm = np.linalg.norm(b, axis=-1)
        b_dir = b / (b_norm[:, None] + 1e-8)
        s2 = np.log(np.e - 1) * np.ones_like(c2)

        self.b_dir = nn.Parameter(torch.from_numpy(b_dir).float())
        self.b_norm = nn.Parameter(torch.from_numpy(b_norm).float())
        self.c2 = nn.Parameter(torch.from_numpy(c2).float())
        self.s2 = nn.Parameter(torch.from_numpy(s2).float())
        self.beta2 = nn.Parameter(torch.Tensor([config.beta]).float())

        self.post_init()

    def unpack_coeffs(
        self,
    ) -> tuple[
        Float[Tensor, "n 3"],
        Float[Tensor, " n"],
        Float[Tensor, " n"],
        Float[Tensor, " 1"],
        Float[Tensor, "n 3"],
        Float[Tensor, " n"],
        Float[Tensor, " n"],
        Float[Tensor, " 1"],
    ]:
        a = F.normalize(self.a_dir, dim=-1) * self.a_norm[:, None]
        s1 = F.softplus(self.s1)
        beta1 = F.softplus(self.beta1)

        b = F.normalize(self.b_dir, dim=-1) * self.b_norm[:, None]
        s2 = F.softplus(self.s2)
        beta2 = F.softplus(self.beta2)

        return a, self.c1, s1, beta1, b, self.c2, s2, beta2

    def forward(self, X: Float[Tensor, "m 3"], accurate_grad: bool = True):
        a, c1, s1, beta1, b, c2, s2, beta2 = self.unpack_coeffs()

        # This is accurate except global betas, which are too slow to update
        if accurate_grad:
            grad_x = grad_LSE(X, a, c1, s1, beta1.detach()) - grad_LSE(
                X, b, c2, s2, beta2.detach()
            )

        A = torch.hstack([a, c1[:, None]])
        B = torch.hstack([b, c2[:, None]])
        X = torch.hstack([X, torch.ones((len(X), 1), dtype=X.dtype, device=X.device)])

        h, _, ws1, h_idx, _ = logsumexp(X, A, s1, beta1)
        g, _, ws2, g_idx, _ = logsumexp(X, B, s2, beta2)

        return (
            h - g,
            grad_x if accurate_grad else (a[h_idx] - b[g_idx]),
            s1,
            s2,
            ws1,
            ws2,
        )

    @torch.no_grad()
    def prune(self, softplus_thr: float):
        self.s1.copy_(torch.where(self.s1 < softplus_thr, -50, self.s1))
        self.s2.copy_(torch.where(self.s2 < softplus_thr, -50, self.s2))


class PruneCallback(TrainerCallback):
    def __init__(self, prune_every: int, softplus_thr: float):
        self.prune_every = prune_every
        self.softplus_thr = softplus_thr

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step != 0 and state.global_step % self.prune_every == 0:
            kwargs["model"].prune(self.softplus_thr)
        return control


class PatchworkTrainer(Trainer):
    def __init__(self, *args, tb_log_dir: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.tb_writer = SummaryWriter(tb_log_dir)

    def compute_loss(
        self, model: Patchwork, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # batch_size 1
        samples = inputs["samples"][0]
        sur_samples = inputs["sur_samples"][0]
        sur_normals = inputs["sur_normals"][0]

        pred_sdfs_surf, pred_normals_surf, s1_surf, s2_surf, ws1_surf, ws2_surf = model(
            sur_samples, accurate_grad=model.config.accurate_grad
        )
        pred_sdfs, _, s1, s2, ws1, ws2 = model(samples, accurate_grad=False)
        loss_mse = torch.abs(pred_sdfs_surf).mean()
        loss_normal = (1 - F.cosine_similarity(pred_normals_surf, sur_normals)).mean()

        pred_occs = F.sigmoid(-pred_sdfs)
        loss_reg = torch.square(double_well_potential(pred_occs)).mean()

        loss_prune = (
            torch.abs(s1_surf).mean()
            + torch.abs(s2_surf).mean()
            + torch.abs(s1).mean()
            + torch.abs(s2).mean()
            + F.relu(1 - torch.abs(ws1_surf)).mean()
            + F.relu(1 - torch.abs(ws2_surf)).mean()
            + F.relu(1 - torch.abs(ws1)).mean()
            + F.relu(1 - torch.abs(ws2)).mean()
        )

        loss = (
            model.config.w_mse * loss_mse
            + model.config.w_normal * loss_normal
            + model.config.w_reg * loss_reg
            + model.config.w_prune * loss_prune
        )

        step = self.state.global_step
        self.tb_writer.add_scalar("loss/mse", loss_mse.item(), step)
        self.tb_writer.add_scalar("loss/normal", loss_normal.item(), step)
        self.tb_writer.add_scalar("loss/reg", loss_reg.item(), step)
        self.tb_writer.add_scalar("loss/prune", loss_prune.item(), step)
        self.tb_writer.add_scalar("loss/total", loss.item(), step)
        return (loss, None) if return_outputs else loss


def eval(func: callable, grid_res=512):
    axis = torch.linspace(-1, 1, grid_res)
    grid_pts = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(
        -1, 3
    )
    group_size = grid_res**2

    sdfs = []
    for pt_group in tqdm(torch.split(grid_pts, group_size)):
        sdf_group = func(pt_group.float().cuda()).detach().cpu()
        sdfs.append(sdf_group)
    sdfs = torch.concat(sdfs)
    sdfs = sdfs.reshape(grid_res, grid_res, grid_res).numpy()

    spacing = 1.0 / grid_res
    V, F, _, _ = marching_cubes(sdfs, 0, spacing=(spacing, spacing, spacing))
    V = 2 * (V - 0.5)
    return V, F


if __name__ == "__main__":
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--exp_dir", type=str, help="Exp folder")
    parser.add_argument("--N", type=int, default=16384, help="Input point cloud size")
    parser.add_argument("--rho", type=int, default=200, help="Rho for geo init")
    parser.add_argument("--w_mse", type=float, default=1.0, help="Weight mse")
    parser.add_argument("--w_normal", type=float, default=1.0, help="Weight normal")
    parser.add_argument("--w_reg", type=float, default=1.0, help="Weight reg")
    parser.add_argument("--w_prune", type=float, default=1.0, help="Weight prune")
    parser.add_argument(
        "--grad_pass_through",
        action="store_true",
        help="Cheap normal approximation with pass through, half the training cost.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=10000, help="Num of iterations")
    parser.add_argument("--batch_size", type=int, default=16384, help="Batch size")
    parser.add_argument("--res", type=int, default=256, help="MC res")
    parser.add_argument("--debug", action="store_true", help="Debug only")
    parser.add_argument("--eval", action="store_true", help="Evaluate only")
    parser.add_argument("--vis", action="store_true", help="Visualize extraction")
    args = parser.parse_args()

    model_name = args.model_name
    tag = f"{model_name}"
    N = args.N
    rho = args.rho
    lr = args.lr
    w_mse = args.w_mse
    w_normal = args.w_normal
    w_reg = args.w_reg
    w_prune = args.w_prune
    grad_pass_through = args.grad_pass_through
    exp_dir = args.exp_dir
    n_steps = args.n_steps
    batch_size = args.batch_size

    gt_path = glob(os.path.join("data/mesh", f"{model_name}.*"))[0]
    V_gt, F_gt = igl.read_triangle_mesh(gt_path)
    V_gt = normalize_aabb(V_gt)

    pc_path = f"data/pc/{model_name}.ply"
    pc_o3d = o3d.io.read_point_cloud(os.path.expandvars(pc_path))
    sur_samples = np.asarray(pc_o3d.points)
    sur_normals = np.asarray(pc_o3d.normals)

    softplus_thr = -4.6

    cfg = PatchworkConfig(
        rho=rho,
        w_mse=w_mse,
        w_normal=w_normal,
        w_reg=w_reg,
        w_prune=w_prune,
        grad_pass_through=grad_pass_through,
    )
    patchwork = Patchwork(cfg, sur_samples[:N], sur_normals[:N])

    # Eval
    if args.eval:
        patchwork: Patchwork = Patchwork.from_pretrained(
            os.path.join(exp_dir, model_name, f"checkpoint-{n_steps}"),
            sur_samples[:N],
            sur_normals[:N],
            config=cfg,
        ).cuda()
        a, c1, s1, beta1, b, c2, s2, beta2 = patchwork.unpack_coeffs()

        def _eval(X: Float[Tensor, "m 3"]):
            A = torch.hstack([a, c1[:, None]])
            B = torch.hstack([b, c2[:, None]])
            X = torch.hstack(
                [X, torch.ones((len(X), 1), dtype=X.dtype, device=X.device)]
            )
            return logsumexp(X, A, s1, beta1)[0] - logsumexp(X, B, s2, beta2)[0]

        V, F = eval(_eval, args.res)

        ps.init()
        ps.register_surface_mesh("Mesh", V, F)
        ps.register_surface_mesh("GT", V_gt, F_gt)
        ps.show()
        exit()

    dataset = SDFDataset(model_name, n_steps, batch_size)

    training_args = TrainingArguments(
        output_dir=os.path.join(exp_dir, model_name),
        per_device_train_batch_size=1,
        max_steps=n_steps,
        learning_rate=lr,
        remove_unused_columns=False,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=n_steps // 10,
    )
    trainer = PatchworkTrainer(
        model=patchwork,
        args=training_args,
        train_dataset=dataset,
        tb_log_dir=os.path.join(exp_dir, model_name, "tb_logs"),
        callbacks=[PruneCallback(prune_every=n_steps // 5, softplus_thr=softplus_thr)],
    )
    out = trainer.train()

    num_params = (trainer.model.s1 > softplus_thr).sum() + (
        trainer.model.s2 > softplus_thr
    ).sum()
    elapsed = out.metrics["train_runtime"]
    meta = {"num_params": num_params.item(), "elapsed": elapsed}
    with open(os.path.join(exp_dir, model_name, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(meta)

    a, c1, s1, beta1, b, c2, s2, beta2 = trainer.model.unpack_coeffs()

    def _eval(X: Float[Tensor, "m 3"]):
        A = torch.hstack([a, c1[:, None]])
        B = torch.hstack([b, c2[:, None]])
        X = torch.hstack([X, torch.ones((len(X), 1), dtype=X.dtype, device=X.device)])
        return logsumexp(X, A, s1, beta1)[0] - logsumexp(X, B, s2, beta2)[0]

    V, F = eval(_eval, args.res)
    save_path = os.path.join(exp_dir, "result_meshes", f"{tag}.obj")
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    igl.write_triangle_mesh(save_path, V, F)

    if args.vis:
        ps.init()
        ps.register_surface_mesh("Mesh", V, F)
        ps.register_surface_mesh("GT", V_gt, F_gt)
        ps.show()
