import math

from fit_geo_3d import eval_power_dist_flipped, Patchwork

import cv2
import numpy as np
from patchwork_query_ext.kdtree import KDTree
import torch
from torch import Tensor
from tqdm import tqdm

from icecream import ic
import polyscope as ps


# Camera code adapted from: https://github.com/NVlabs/neuralangelo/blob/94390b64683c067c620d9e075224ccfe582647d0/projects/nerf/utils/camera.py
#   and https://github.com/fraunhoferhhi/neural-deferred-shading


def intersect_with_sphere(center, ray_unit, radius=math.sqrt(3)):
    ctc = (center * center).sum(dim=-1, keepdim=True)  # [...,1]
    ctv = (center * ray_unit).sum(dim=-1, keepdim=True)  # [...,1]
    b2_minus_4ac = ctv**2 - (ctc - radius**2)

    dist_near = torch.where(b2_minus_4ac < 0, -1, -ctv - b2_minus_4ac.sqrt())
    dist_far = torch.where(b2_minus_4ac < 0, -1, -ctv + b2_minus_4ac.sqrt())
    return dist_near, dist_far


def to_hom(X):
    # Get homogeneous coordinates of the input.
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


# Basic operations of transforming 3D points between world/camera/image coordinates.
def world2cam(X, pose):  # [B,N,3]
    X_hom = to_hom(X)
    return X_hom @ pose.transpose(-1, -2)


def cam2img(X, cam_intr):
    return X @ cam_intr.transpose(-1, -2)


def img2cam(X, cam_intr):
    return X @ cam_intr.inverse().transpose(-1, -2)


class Camera:
    """Camera in OpenCV format.

    Args:
        K (tensor): Camera matrix with intrinsic parameters (3x3)
        R (tensor): Rotation matrix (3x3)
        t (tensor): translation vector (3)
        device (torch.device): Device where the matrices are stored
    """

    def __init__(self, K, R, t, device="cpu"):
        self.K = K.to(device) if torch.is_tensor(K) else torch.FloatTensor(K).to(device)
        self.R = R.to(device) if torch.is_tensor(R) else torch.FloatTensor(R).to(device)
        self.t = t.to(device) if torch.is_tensor(t) else torch.FloatTensor(t).to(device)
        self.device = device

    def to(self, device="cpu"):
        self.K = self.K.to(device)
        self.R = self.R.to(device)
        self.t = self.t.to(device)
        self.device = device
        return self

    @property
    def center(self):
        return -self.R.t() @ self.t

    @property
    def P(self):
        return self.K @ torch.cat([self.R, self.t.unsqueeze(-1)], dim=-1)


class Pose:
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4]).
    Each [3,4] camera pose takes the form of [R|t].
    """

    def __call__(self, R=None, t=None):
        # Construct a camera pose from the given R and/or t.
        assert R is not None or t is not None
        if R is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1)  # [...,3,4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False):
        # Invert a camera pose.
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # Compose a sequence of poses together.
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new

    def scale_center(self, pose, scale):
        """Scale the camera center from the origin.
        0 = R@c+t --> c = -R^T@t (camera center in world coordinates)
        0 = R@(sc)+t' --> t' = -R@(sc) = -R@(-R^T@st) = st
        """
        R, t = pose[..., :3], pose[..., 3:]
        pose_new = torch.cat([R, t * scale], dim=-1)
        return pose_new


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom @ pose_inv.transpose(-1, -2)


class View:
    """A View is a combination of camera and image(s).

    Args:
        color (tensor): RGB color image (WxHx3)
        mask (tensor): Object mask (WxHx1)
        camera (Camera): Camera associated with this view
        device (torch.device): Device where the images and camera are stored
    """

    def __init__(self, H, W, camera, device="cpu"):
        self.H = H
        self.W = W
        self.camera: Camera = camera.to(device)
        self.device = device

    @property
    def resolution(self):
        return (self.H, self.W)

    def scale(self, inverse_factor):
        """Scale the view by a factor.

        This operation is NOT differentiable in the current state as
        we are using opencv.

        Args:
            inverse_factor (float): Inverse of the scale factor (e.g. to halve the image size, pass `2`)
        """

        scaled_height = self.H // inverse_factor
        scaled_width = self.W // inverse_factor

        scale_x = scaled_width / self.W
        scale_y = scaled_height / self.H

        self.camera.K = (
            torch.FloatTensor(np.diag([scale_x, scale_y, 1])).to(self.device)
            @ self.camera.K
        )

    def transform(self, A, A_inv=None):
        """Transform the view pose with an affine mapping.

        Args:
            A (tensor): Affine matrix (4x4)
            A_inv (tensor, optional): Inverse of the affine matrix A (4x4)
        """

        if not torch.is_tensor(A):
            A = torch.from_numpy(A)

        if A_inv is not None and not torch.is_tensor(A_inv):
            A_inv = torch.from_numpy(A_inv)

        A = A.to(self.device, dtype=torch.float32)
        if A_inv is not None:
            A_inv = A_inv.to(self.device, dtype=torch.float32)

        if A_inv is None:
            A_inv = torch.inverse(A)

        # Transform camera extrinsics according to  [R'|t'] = [R|t] * A_inv.
        # We compose the projection matrix and decompose it again, to correctly
        # propagate scale and shear related factors to the K matrix,
        # and thus make sure that R is a rotation matrix.
        R = self.camera.R @ A_inv[:3, :3]
        t = self.camera.R @ A_inv[:3, 3] + self.camera.t
        P = torch.zeros((3, 4), device=self.device)
        P[:3, :3] = self.camera.K @ R
        P[:3, 3] = self.camera.K @ t
        K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P.cpu().detach().numpy())
        c = c[:3, 0] / c[3]
        t = -R @ c

        # ensure unique scaling of K matrix
        K = K / K[2, 2]

        self.camera.K = torch.from_numpy(K).to(self.device, dtype=self.camera.K.dtype)
        self.camera.R = torch.from_numpy(R).to(self.device, dtype=self.camera.R.dtype)
        self.camera.t = torch.from_numpy(t).to(self.device, dtype=self.camera.t.dtype)

    def generate_ray(self):
        H, W = self.resolution
        with torch.no_grad():
            # Compute image coordinate grid.
            y_range = torch.arange(H, dtype=torch.float32, device=self.device).add_(0.5)
            x_range = torch.arange(W, dtype=torch.float32, device=self.device).add_(0.5)
            Y, X = torch.meshgrid(y_range, x_range, indexing="ij")  # [H,W]
            xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]

        pose = Pose()(self.camera.R, self.camera.t)
        grid_3D = img2cam(to_hom(xy_grid), self.camera.K)  # [HW,3]/[HW,3]
        center_3D = torch.zeros_like(grid_3D)  # [HW,3]/[HW,3]
        # Transform from camera to world coordinates.
        grid_3D = cam2world(grid_3D, pose)  # [HW,3]/[HW,3]
        center_3D = cam2world(center_3D, pose)  # [HW,3]/[HW,3]
        ray = grid_3D - center_3D  # [HW,3]
        return center_3D, torch.nn.functional.normalize(ray)

    def generate_ray_masked(self):
        center_3D, ray = self.generate_ray()
        mask = (
            self.mask.reshape(
                -1,
            )
            > 0
        )
        return center_3D[mask], ray[mask]


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return (v, v, v)

    h *= 6
    i = int(math.floor(h))
    f = h - i  # fractional part of h
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    if i == 0:
        return np.array([v, t, p])
    elif i == 1:
        return np.array([q, v, p])
    elif i == 2:
        return np.array([p, v, t])
    elif i == 3:
        return np.array([p, q, v])
    elif i == 4:
        return np.array([t, p, v])
    else:
        return np.array([v, p, q])


def hsv_to_rgb_torch(h, s, v):
    """
    h, s, v: 1D tensors of shape (N,), values in [0, 1]
    Returns: (N, 3) tensor of RGB values in [0, 1]
    """
    h = h * 6.0
    i = torch.floor(h)
    f = h - i  # fractional part
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    i = i.long() % 6  # handle h == 1.0 → i == 6

    # Stack candidates for each sextant: shape (6, N, 3)
    candidates = torch.stack(
        [
            torch.stack([v, t, p], dim=-1),
            torch.stack([q, v, p], dim=-1),
            torch.stack([p, v, t], dim=-1),
            torch.stack([p, q, v], dim=-1),
            torch.stack([t, p, v], dim=-1),
            torch.stack([v, p, q], dim=-1),
        ],
        dim=0,
    )

    # Select the right candidate per element
    rgb = candidates.gather(0, i.view(1, -1, 1).expand(1, -1, 3)).squeeze(0)

    # s == 0 → grayscale (v, v, v)
    gray = v.unsqueeze(-1).expand(-1, 3)
    rgb = torch.where(s.unsqueeze(-1) == 0, gray, rgb)

    return rgb


class PatchworkRender:
    def __init__(self, device, near=1e-3, far=2.0):
        self.device = device
        self.near = near
        self.far = far

    @torch.no_grad
    def render(self, view: View, patchwork: Patchwork):
        # Generate rays
        ray_o, ray_dir = view.generate_ray()
        # Consider adjust sampling step based their difference
        t_near, t_far = intersect_with_sphere(ray_o, ray_dir)
        t_near = self.near * torch.ones_like(t_near)
        t_far = self.far * torch.ones_like(t_far)

        h, w = view.resolution[0], view.resolution[1]

        _, s1, ga, ra2, _, s2, gb, rb2 = patchwork.unpack_compressed_coeffs()

        tree_a = KDTree(ga, ra2, s1)
        tree_b = KDTree(gb, rb2, s2)

        def _eval(X: Tensor):
            h_idx = tree_a.query_closest(X)
            g_idx = tree_b.query_closest(X)

            sdf = eval_power_dist_flipped(
                X, ga[h_idx], ra2[h_idx]
            ) - eval_power_dist_flipped(X, gb[g_idx], rb2[g_idx])
            return sdf, torch.nn.functional.normalize(ga[h_idx] - gb[g_idx])

        # Let's brutal force raymarching, then consider speed up
        query_pos, query_normal, query_mask = self.ray_marching(
            ray_o, ray_dir, t_near, t_far, _eval
        )
        query_mask = query_mask.reshape(h, w)
        # query_normal = query_normal @ view.camera.R

        normal_map = torch.zeros((h, w, 3), dtype=torch.float32, device=view.device)
        normal_map[query_mask] = 0.5 * (query_normal + 1)
        normal_map = torch.concat([normal_map, query_mask.reshape(h, w, 1)], dim=-1)

        g_idx = tree_b.query_closest(query_pos)
        cell_map = torch.zeros((h, w, 3), dtype=torch.float32, device=view.device)

        light_dir = torch.Tensor([1.0, 1.0, 1.0]).to(query_normal.device)
        light_dir = light_dir / (1e-8 + torch.linalg.norm(light_dir))
        shading_dp = torch.einsum("nd,d->n", query_normal, light_dir)

        hue = (g_idx * 0.61803398) % 1
        saturation = 0.5 + 0.5 * torch.rand_like(hue)
        value = 0.75 * torch.clip(shading_dp, 0.0, 1.0) + 0.25

        color = hsv_to_rgb_torch(hue, saturation, value)
        cell_map[query_mask] = color
        cell_map = torch.concat([cell_map, query_mask.reshape(h, w, 1)], dim=-1)

        return query_pos, query_normal, query_mask, normal_map, cell_map

    def ray_marching(
        self,
        rays_o,
        rays_dir,
        ts_near,
        ts_far,
        decode,
        step_size=512,
        secant_step=3,
        batch_size=64,
    ):
        # b x n x 3
        rays_o = rays_o.reshape(batch_size, -1, 3)
        # b x n x 3
        rays_dir = rays_dir.reshape(batch_size, -1, 3)
        # b x n
        ts_near = ts_near.reshape(batch_size, -1)
        # b x n
        ts_far = ts_far.reshape(batch_size, -1)

        intervals = torch.linspace(0, 1, step_size).to(rays_o)

        # b x n
        proposals_mask = ts_near > 0

        results_positions = []
        results_normals = []
        result_masks = []
        # trade query calls with indexing complexity
        for b in tqdm(range(batch_size)):
            # n'
            mask_batch = proposals_mask[b]
            # n' x 3
            rays_o_batch = rays_o[b][mask_batch]
            # n' x 3
            rays_dir_batch = rays_dir[b][mask_batch]
            # n'
            ts_near_batch = ts_near[b][mask_batch]
            # n'
            ts_far_batch = ts_far[b][mask_batch]

            # step_size x n'
            zs_batch = (
                ts_near_batch[None, :]
                + intervals[:, None] * (ts_far_batch - ts_near_batch)[None, :]
            )
            # step_size x n' x 3
            proposals_batch = (
                rays_o_batch[None, ...]
                + zs_batch[..., None] * rays_dir_batch[None, ...]
            )

            # step_size x n'
            sdf_batch = decode(
                proposals_batch.reshape(-1, 3),
            )[0].reshape(step_size, -1)

            # invalid if first query has sdf < 0 (i.e. view from inside)
            invalid_first_sdf = sdf_batch[0, :] < 0
            # invalid if no negative
            no_negative = (sdf_batch < 0).sum(0) == 0

            invalid_mask = torch.logical_or(invalid_first_sdf, no_negative)

            # update mask
            mask_batch[mask_batch.nonzero()[invalid_mask]] = False
            result_masks.append(mask_batch)

            if mask_batch.sum() == 0:
                continue

            # prepare for raymarching
            # n''
            valid_mask = torch.logical_not(invalid_mask)
            # n''
            valid_indices = valid_mask.nonzero().reshape(-1)

            # choose the earliest sign change
            # (step_size - 1) x n'
            sign_test = sdf_batch[:-1, :] * sdf_batch[1:, :]
            # (step_size - 1) x n''
            sign_cost = (
                torch.sign(sign_test.index_select(dim=1, index=valid_indices))
                * torch.arange(step_size, 1, step=-1).to(sign_test)[:, None]
            )
            # This happens, although super rare...
            if sign_cost.numel() > 0:
                sign_cost[sign_cost == 0] = sign_cost.min()
            # n''
            min_indices = sign_cost.min(dim=0).indices
            sdf_batch_valid = sdf_batch.index_select(dim=1, index=valid_indices)

            # Run Secant method (faster than bisection, more efficient than Newton)
            rays_o_batch_valid = rays_o_batch[valid_mask]
            rays_dir_batch_valid = rays_dir_batch[valid_mask]

            y_0 = torch.gather(
                sdf_batch_valid, dim=0, index=min_indices[None, :]
            ).reshape(-1, 1)
            y_1 = torch.gather(
                sdf_batch_valid, dim=0, index=min_indices[None, :] + 1
            ).reshape(-1, 1)

            zs_batch_valid = zs_batch.index_select(dim=1, index=valid_indices)
            z_0 = torch.gather(
                zs_batch_valid, dim=0, index=min_indices[None, :]
            ).reshape(-1, 1)
            z_1 = torch.gather(
                zs_batch_valid, dim=0, index=min_indices[None, :] + 1
            ).reshape(-1, 1)

            def f_sdf(z):
                sdf_eval = decode(rays_o_batch_valid + z * rays_dir_batch_valid)[
                    0
                ].reshape(-1, 1)
                return sdf_eval

            z_opt = self.run_secant(z_0, z_1, y_0, y_1, f_sdf, secant_step=secant_step)
            query_opt = rays_o_batch_valid + z_opt * rays_dir_batch_valid
            _, query_normal = decode(query_opt)

            results_positions.append(query_opt)
            results_normals.append(query_normal)

        if len(results_positions) == 0:
            return {
                torch.zeros((0, 3)).to(rays_o.device),
                torch.zeros((0, 3)).to(rays_o.device),
                proposals_mask.reshape(
                    -1,
                ),
            }

        results_positions = torch.vstack(results_positions)
        results_normals = torch.vstack(results_normals)
        result_masks = torch.stack(result_masks)

        return (
            results_positions,
            results_normals,
            proposals_mask.reshape(
                -1,
            ),
        )

    def run_secant(self, x_0, x_1, y_0, y_1, f_sdf, secant_step=8, eps=1e-5):
        x_2 = x_1 - y_1 * (x_1 - x_0) / (y_1 - y_0 + eps)
        y_2 = f_sdf(x_2)

        split_mask = y_2 > 0
        x_0 = torch.where(split_mask, x_0, x_2)
        x_1 = torch.where(split_mask, x_2, x_1)
        y_0 = torch.where(split_mask, y_0, y_2)
        y_1 = torch.where(split_mask, y_2, y_1)

        if secant_step <= 0:
            return torch.where(y_0.abs() < y_1.abs(), x_0, x_1)
        else:
            return self.run_secant(x_0, x_1, y_0, y_1, f_sdf, secant_step - 1)


if __name__ == "__main__":
    H = 512
    W = H
    fov = 55
    fx = 0.5 * (H / np.tan(np.deg2rad(fov) / 2))
    fy = fx
    K = np.array([[fx, 0, H // 2], [0, fy, W // 2], [0, 0, 1]])

    w2c = np.array(
        [
            [-4.37113880e-08, 1.00000000e00, 8.46357033e-09, -9.99998858e-02],
            [4.47213535e-01, 2.71183711e-08, -8.94427190e-01, 2.44721377e00],
            [-8.94427190e-01, -3.53116314e-08, -4.47213535e-01, 4.07770864e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    R = w2c[:3, :3]
    t = w2c[:3, 3]

    camera = Camera(K, R, t, device="cuda")
    view = View(H, W, camera, device="cuda")

    save_path = "exp_N_16777216/scale/wukong/model.pth"

    N = 16777216
    model = Patchwork(np.empty((N, 3)), np.empty((N, 3)))
    model.load_state_dict(torch.load(save_path))
    model = model.to("cuda")

    A, _, ga, ra2, B, _, gb, rb2 = model.unpack_compressed_coeffs()

    ga_np = ga.detach().cpu().numpy()
    gb_np = gb.detach().cpu().numpy()

    ps.init()
    fy = float(K[1, 1])
    intr = ps.CameraIntrinsics(
        fov_vertical_deg=float(np.degrees(2 * np.arctan(H / (2 * fy)))),
        aspect=float(W) / float(H),
    )
    gl = (
        np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32) @ w2c
    )  # OpenCV -> OpenGL frame
    cam = ps.register_camera_view(
        "render camera", ps.CameraParameters(intr, ps.CameraExtrinsics(mat=gl))
    )

    # ps.register_point_cloud("ga_np", ga_np, point_render_mode="quad")
    ps.register_point_cloud("gb_np", gb_np, point_render_mode="quad")

    renderer = PatchworkRender("cuda")
    query_pos, query_normal, query_mask, normal_map, cell_map = renderer.render(
        view, model
    )

    normal_map = normal_map.detach().cpu().numpy()[..., :-1]
    cell_map = cell_map.detach().cpu().numpy()[..., :-1]

    # ps.register_point_cloud("ga_np", ga_np, point_render_mode="quad")
    ps.register_point_cloud("gb_np", gb_np, point_render_mode="quad")
    ps.register_point_cloud(
        "query_pos", query_pos.detach().cpu(), point_render_mode="quad"
    ).add_vector_quantity("query_normal", query_normal.detach().cpu(), enabled=True)
    cam.add_color_image_quantity(
        "normal_map", normal_map, enabled=True, show_in_camera_billboard=True
    )
    cam.add_color_image_quantity(
        "cell_map", cell_map, enabled=True, show_in_camera_billboard=True
    )
    ps.show()
