import argparse
from dataclasses import dataclass, field
from glob import glob
import os

from triton_kernel import logsumexp

import igl
from jaxtyping import Float
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedConfig, PreTrainedModel, Trainer, TrainingArguments

from icecream import ic
import polyscope as ps


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


class SDFDataset(Dataset):
    def __init__(self, model_name: str, n_steps, n_samples):
        super().__init__()

        pc_path = f"data/pc/{model_name}.ply"
        pc_o3d = o3d.io.read_point_cloud(os.path.expandvars(pc_path))
        self.sur_samples = np.asarray(pc_o3d.points)[:n_samples]
        self.sur_normals = np.asarray(pc_o3d.normals)[:n_samples]

        sdf_path = f"data/sdf/{model_name}.npz"
        data = np.load(sdf_path)

        self.samples = data["samples"]
        self.sample_sdfs = data["sample_sdfs"]

        self.n_steps = n_steps
        self.n_samples_per_batch = n_samples
        self.n_batches = len(self.samples) // n_samples

    def __len__(self):
        return self.n_steps

    def __getitem__(self, index):
        prefix = (index % self.n_batches) * self.n_samples_per_batch
        idx = np.arange(self.n_samples_per_batch) + prefix
        samples = self.samples[idx]
        sample_sdfs = self.sample_sdfs[idx]

        return {
            "sur_samples": self.sur_samples.astype(np.float32),
            "sur_normals": self.sur_normals.astype(np.float32),
            "samples": samples.astype(np.float32),
            "sample_sdfs": sample_sdfs.astype(np.float32),
        }


class PatchworkConfig(PreTrainedConfig):
    model_type = "patchwork"

    def __init__(self, rho: float = 200.0, beta: float = 75.0):
        super().__init__()
        self.rho = rho
        self.beta = beta


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

    def forward(self, X: Float[Tensor, "m 3"]):
        a, c1, s1, beta1, b, c2, s2, beta2 = self.unpack_coeffs()

        A = torch.hstack([a, c1[:, None]])
        B = torch.hstack([b, c2[:, None]])
        X = torch.hstack([X, torch.ones((len(X), 1), dtype=X.dtype, device=X.device)])

        h, _, ws1, h_idx, _ = logsumexp(X, A, s1, beta1)
        g, _, ws2, g_idx, _ = logsumexp(X, B, s2, beta2)

        return h - g, a[h_idx] - b[g_idx], s1, s2, ws1, ws2


@dataclass
class PatchworkTrainingArguments(TrainingArguments):
    w_mse: float = field(default=1.0, metadata={"help": "Loss MSE weight."})
    w_normal: float = field(default=1.0, metadata={"help": "Loss Normal weight."})
    w_reg: float = field(default=1.0, metadata={"help": "Loss Reg weight."})
    w_prune: float = field(default=10.0, metadata={"help": "Loss Prune weight."})


class PatchworkTrainer(Trainer):
    def compute_loss(
        self, model: Patchwork, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # batch_size 1
        samples = inputs["samples"][0]
        sur_samples = inputs["sur_samples"][0]
        sur_normals = inputs["sur_normals"][0]

        pred_sdfs_surf, pred_normals_surf, s1_surf, s2_surf, ws1_surf, ws2_surf = model(
            sur_samples
        )
        pred_sdfs, _, s1, s2, ws1, ws2 = model(samples)
        loss_mse = self.args.w_mse * torch.abs(pred_sdfs_surf).mean()
        loss_normal = (
            self.args.w_normal
            * (1 - F.cosine_similarity(pred_normals_surf, sur_normals)).mean()
        )

        pred_occs = F.sigmoid(-pred_sdfs)
        loss_reg = (
            self.args.w_reg * torch.square(double_well_potential(pred_occs)).mean()
        )

        loss_prune = (
            self.args.w_prune
            * (
                torch.abs(s1_surf).mean()
                + torch.abs(s2_surf).mean()
                + torch.abs(s1).mean()
                + torch.abs(s2).mean()
            )
            + F.relu(1 - torch.abs(ws1_surf)).mean()
            + F.relu(1 - torch.abs(ws2_surf)).mean()
            + F.relu(1 - torch.abs(ws1)).mean()
            + F.relu(1 - torch.abs(ws2)).mean()
        )

        loss = loss_mse + loss_normal + loss_reg + loss_prune
        outputs = {
            "loss_mse": loss_mse.detach().item(),
            "loss_normal": loss_normal.detach().item(),
            "loss_reg": loss_reg.detach().item(),
            "loss_prune": loss_prune.detach().item(),
        }
        return (loss, outputs) if return_outputs else loss


def eval(model: Patchwork, grid_res=512):
    axis = torch.linspace(-1, 1, grid_res)
    grid_pts = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(
        -1, 3
    )
    group_size = grid_res**2

    sdfs = []
    for pt_group in tqdm(torch.split(grid_pts, group_size)):
        sdf_group = model(pt_group.float().cuda())[0].detach().cpu()
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
    parser.add_argument("--N", type=int, default=16384, help="Evaluate only")
    parser.add_argument("--rho", type=int, default=200, help="Rho for geo init")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--w_mse", type=float, default=1.0, help="Weight mse")
    parser.add_argument("--w_normal", type=float, default=1.0, help="Weight normal")
    parser.add_argument("--w_reg", type=float, default=1.0, help="Weight reg")
    parser.add_argument("--w_prune", type=float, default=10.0, help="Weight prune")
    parser.add_argument("--res", type=int, default=256, help="MC res")
    parser.add_argument("--vis", action="store_true", help="Evaluate only")
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
    exp_dir = args.exp_dir

    gt_path = glob(os.path.join("data/mesh", f"{model_name}.*"))[0]
    V_gt, F_gt = igl.read_triangle_mesh(gt_path)
    V_gt = normalize_aabb(V_gt)

    pc_path = f"data/pc/{model_name}.ply"
    pc_o3d = o3d.io.read_point_cloud(os.path.expandvars(pc_path))
    sur_samples = np.asarray(pc_o3d.points)
    sur_normals = np.asarray(pc_o3d.normals)

    D = 3
    softplus_thr = -4.6

    cfg = PatchworkConfig(rho)
    patchwork = Patchwork(cfg, sur_samples[:N], sur_normals[:N])

    n_steps = 10000
    batch_size = 16384
    dataset = SDFDataset(model_name, n_steps, batch_size)

    training_args = PatchworkTrainingArguments(
        output_dir=os.path.join(exp_dir, model_name),
        per_device_train_batch_size=1,
        max_steps=n_steps,
        learning_rate=lr,
        remove_unused_columns=False,
        w_mse=w_mse,
        w_normal=w_normal,
        w_reg=w_reg,
        w_prune=w_prune,
    )
    trainer = PatchworkTrainer(
        model=patchwork, args=training_args, train_dataset=dataset
    )
    trainer.train()

    V, F = eval(trainer.model, args.res)
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
