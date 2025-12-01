"""
Torch-based amplitude solver for the eikonal equation (piecewise-constant n).

We trace rays (streamlines of s = ∇φ/|∇φ|) and approximate amplitude via
ray density, applying Snell/refraction jumps at interfaces. This favors
stability and symmetry over exact transport integration.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

# Global defaults for the torch tracer; tweak here for quick experimentation.
# - DEFAULT_TORCH_STEP: Backtrace step in pixels. Smaller -> more steps, smoother
#   advection, but slower. Typical 0.5–0.9. Too small just burns cycles.
DEFAULT_TORCH_STEP = 0.75
# - DEFAULT_TORCH_JITTER: Sub-pixel jitter on seed positions to break grid bias.
#   0.0–0.5 is reasonable. Higher = more isotropy, but noisier.
DEFAULT_TORCH_JITTER = 0.5
# - DEFAULT_TORCH_SUBSAMPLE: Downsample factor for the tracer grid. 1 = full res.
#   >1 speeds up but loses fidelity. Keep 1 unless you need a speed hack.
DEFAULT_TORCH_SUBSAMPLE = 1
# - DEFAULT_TORCH_TARGET_RAYS: Budget to thin dense seed bands. Higher = more rays,
#   better density estimates, slower. Typical 4k–16k depending on scene size.
DEFAULT_TORCH_TARGET_RAYS = 16000
# - DEFAULT_TORCH_BLUR_SIZE / SIGMA: Smoothing of splatted density. 5 with binomial
#   weights is a mild blur. Set to 1 to disable blur (sharper, noisier). Larger
#   blurs soften caustics. SIGMA=None uses binomial for size=5; otherwise Gaussian
#   with sigma≈size/3 by default.
DEFAULT_TORCH_BLUR_SIZE = 5
DEFAULT_TORCH_BLUR_SIGMA: float | None = None
# - DEFAULT_TORCH_INTEGRATE_TRANSPORT: If True, integrates the transport term
#   along rays in addition to density. Off by default; turning on can reduce noise
#   when ray counts are low at the cost of extra math per step.
DEFAULT_TORCH_INTEGRATE_TRANSPORT = False

def _bilinear_sample_np(grid: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinear sample a 2D numpy grid at floating-point coords."""
    h, w = grid.shape
    y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy1 = y - y0
    wx1 = x - x0
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1
    return (
        grid[y0, x0] * wy0 * wx0 +
        grid[y0, x1] * wy0 * wx1 +
        grid[y1, x0] * wy1 * wx0 +
        grid[y1, x1] * wy1 * wx1
    )


def _resample_to_shape(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Resample a 2D grid to the given shape using bilinear interpolation.

    This treats the input as defined on a regular [0, H-1]×[0, W-1] lattice
    and samples it on an output lattice of size (out_h, out_w).
    """
    h, w = grid.shape
    if h == out_h and w == out_w:
        return grid.astype(np.float64, copy=False)

    y = np.linspace(0.0, h - 1.0, out_h, dtype=np.float64)
    x = np.linspace(0.0, w - 1.0, out_w, dtype=np.float64)
    Y, X = np.meshgrid(y, x, indexing="ij")
    grid64 = grid.astype(np.float64, copy=False)
    return _bilinear_sample_np(grid64, Y, X)


def _to_torch(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32, non_blocking=True)


def _bilinear_sample(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinear sample a 2D grid at floating-point coords (y, x)."""
    if grid.dim() == 2:
        grid = grid.unsqueeze(0)  # (1, H, W)
    c, h, w = grid.shape
    y = coords[:, 0].clamp(0, h - 1)
    x = coords[:, 1].clamp(0, w - 1)

    y0 = torch.floor(y)
    x0 = torch.floor(x)
    y1 = torch.minimum(y0 + 1, torch.tensor(h - 1, device=grid.device))
    x1 = torch.minimum(x0 + 1, torch.tensor(w - 1, device=grid.device))

    y0w = (y1 - y).unsqueeze(1)
    y1w = (y - y0).unsqueeze(1)
    x0w = (x1 - x).unsqueeze(1)
    x1w = (x - x0).unsqueeze(1)

    def gather(y_idx, x_idx):
        return grid[:, y_idx.long(), x_idx.long()].transpose(0, 1)  # (N, C)

    q00 = gather(y0, x0)
    q01 = gather(y0, x1)
    q10 = gather(y1, x0)
    q11 = gather(y1, x1)

    top = q00 * x0w + q01 * x1w
    bottom = q10 * x0w + q11 * x1w
    out = top * y0w + bottom * y1w  # (N, C)
    return out.squeeze(1) if c == 1 else out


def _snell_transmit(dir_vec: torch.Tensor, n_in: torch.Tensor, n_out: torch.Tensor, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Snell's law to compute transmitted direction and amplitude factor.
    Returns (new_dir, log_T_amplitude).
    """
    nrm = normal / torch.linalg.norm(normal, dim=1, keepdim=True).clamp_min(1e-6)
    cos_theta_in = (-dir_vec * nrm).sum(dim=1).clamp(-1.0, 1.0)
    eta = n_in / n_out
    sin2_theta_t = eta * eta * (1.0 - cos_theta_in * cos_theta_in)
    tir = sin2_theta_t > 1.0
    cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin2_theta_t, min=0.0))

    new_dir = eta.unsqueeze(1) * dir_vec + (eta * cos_theta_in - cos_theta_t).unsqueeze(1) * nrm
    new_dir = new_dir / torch.linalg.norm(new_dir, dim=1, keepdim=True).clamp_min(1e-6)

    T = (n_in * cos_theta_t) / (n_out * cos_theta_in.clamp_min(1e-6))
    log_T = torch.zeros_like(T)  # amplitude handled by ray density; keep neutral transmission

    if tir.any():
        refl_dir = dir_vec - 2.0 * (dir_vec * nrm).sum(dim=1, keepdim=True) * nrm
        new_dir[tir] = refl_dir[tir] / torch.linalg.norm(refl_dir[tir], dim=1, keepdim=True).clamp_min(1e-6)
        log_T[tir] = 0.0
    return new_dir, log_T


def trace_amplitude_torch(
    phi: np.ndarray,
    n_field: np.ndarray,
    source_mask: np.ndarray,
    step_size: float = DEFAULT_TORCH_STEP,
    seed_stride: int = 1,
    jitter: float = DEFAULT_TORCH_JITTER,
    integrate_transport: bool = DEFAULT_TORCH_INTEGRATE_TRANSPORT,
    max_steps: int | None = None,
    device: str | torch.device = "cpu",
    subsample: int = DEFAULT_TORCH_SUBSAMPLE,
    target_rays: int = DEFAULT_TORCH_TARGET_RAYS,
    blur_size: int = DEFAULT_TORCH_BLUR_SIZE,
    blur_sigma: float | None = DEFAULT_TORCH_BLUR_SIGMA,
) -> np.ndarray:
    """
    Trace rays and estimate amplitude via ray density (piecewise-constant n).

    Args:
        phi: Travel time field (H, W)
        n_field: Refractive index (H, W)
        source_mask: Boolean mask of sources (H, W)
        step_size: Step size in pixels
        seed_stride: Seed rays every N pixels (fallback)
        jitter: Random jitter added to seed positions
        integrate_transport: If True, damp intensity by exp(-∫(∇·s + s·∇ln n) ds).
            Off by default because the ray-density splatting already captures
            geometric spreading; enable to reduce noise when ray counts are low.
        max_steps: Optional cap on steps
        device: 'cpu', 'cuda', or 'mps'
        subsample: If >1, run the tracer on a coarse grid and upsample A
            back to the original resolution for speed.
        target_rays: Thin dense source-ring seeds toward this budget.
        blur_size: Gaussian blur kernel size (odd). If <=1, disable blur.
        blur_sigma: Optional sigma for blur. If None, a binomial 5x5 is used
            when blur_size==5; otherwise sigma≈blur_size/3.
    """
    device = torch.device(device)
    subsample = max(1, int(subsample))

    # Optional coarse grid to reduce work.
    h_full, w_full = phi.shape
    if subsample > 1:
        hc = max(1, int(np.ceil(h_full / subsample)))
        wc = max(1, int(np.ceil(w_full / subsample)))
        phi_work = _resample_to_shape(phi, hc, wc)
        n_work = _resample_to_shape(n_field, hc, wc)
        src_f = _resample_to_shape(source_mask.astype(np.float64), hc, wc)
        src_work = src_f > 0.5
        step_size = float(step_size) / float(subsample)
    else:
        phi_work = phi
        n_work = n_field
        src_work = source_mask

    h, w = phi_work.shape
    if max_steps is None:
        max_steps = int(1.25 * max(h, w) / step_size)

    phi_t = _to_torch(phi_work, device)
    n_t = _to_torch(n_work, device)
    src_t = torch.from_numpy(src_work.astype(np.float32)).to(device)

    # Light smoothing of phi to stabilize curvature estimates
    kernel3 = torch.tensor([[1., 2., 1.],
                            [2., 4., 2.],
                            [1., 2., 1.]], device=device, dtype=torch.float32)
    kernel3 = kernel3 / kernel3.sum()
    phi_t = F.conv2d(phi_t.unsqueeze(0).unsqueeze(0), kernel3.view(1, 1, 3, 3), padding=1).squeeze(0).squeeze(0)

    gy, gx = torch.gradient(phi_t)
    mag = torch.sqrt(gx * gx + gy * gy).clamp_min(1e-6)
    sx = gx / mag
    sy = gy / mag
    ny, nx = torch.gradient(n_t)
    ln_n = torch.log(torch.clamp(n_t, min=1e-6))
    ln_n_y, ln_n_x = torch.gradient(ln_n)
    div_s = torch.gradient(sx)[1] + torch.gradient(sy)[0]
    dir_grad_ln_n = ln_n_x * sx + ln_n_y * sy
    transport_field = div_s + dir_grad_ln_n

    # Seed rays on a thin band outside the source
    dist_out = distance_transform_edt((~src_work).astype(np.uint8)).astype(np.float32)
    band_mask = (dist_out > 0.5) & (dist_out < 2.5)
    band_y, band_x = np.nonzero(band_mask)
    if band_y.size == 0:
        yy, xx = np.mgrid[0:h:seed_stride, 0:w:seed_stride]
        band_y, band_x = yy.ravel(), xx.ravel()
    # Thin dense rings to keep ray count reasonable on large grids
    target_rays = max(1, int(target_rays))
    if band_y.size > target_rays:
        stride = int(np.ceil(np.sqrt(band_y.size / target_rays)))
        band_y = band_y[::stride]
        band_x = band_x[::stride]
    seeds_np = np.stack([band_y.astype(np.float32), band_x.astype(np.float32)], axis=1)
    seeds = torch.from_numpy(seeds_np).to(device)

    # Anti-alias thin rings: duplicate seeds with sub-pixel offsets when few samples
    if seeds.shape[0] < 2048:
        offsets = torch.tensor(
            [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
            device=device,
            dtype=seeds.dtype,
        )
        seeds = (seeds[:, None, :] + offsets[None, :, :]).reshape(-1, 2)

    if jitter > 0:
        seeds = seeds + (torch.rand_like(seeds) - 0.5) * (2 * jitter)
        seeds[:, 0] = seeds[:, 0].clamp(0, h - 1)
        seeds[:, 1] = seeds[:, 1].clamp(0, w - 1)

    pos = seeds  # (N,2)
    dir_vec = torch.stack([
        _bilinear_sample(sy, pos),
        _bilinear_sample(sx, pos)
    ], dim=1)
    dir_vec = dir_vec / torch.linalg.norm(dir_vec, dim=1, keepdim=True).clamp_min(1e-6)

    weight = torch.ones(len(pos), device=device)  # intensity weights
    alive = torch.ones(len(pos), device=device, dtype=torch.bool)

    acc_w = torch.zeros((h, w), device=device)

    # Deposit initial weight at seeds to avoid starving the boundary band
    if len(pos) > 0:
        y = pos[:, 0].clamp(0, h - 1)
        x = pos[:, 1].clamp(0, w - 1)
        y0 = torch.floor(y)
        x0 = torch.floor(x)
        y1 = torch.clamp(y0 + 1, max=h - 1)
        x1 = torch.clamp(x0 + 1, max=w - 1)
        wy1 = y - y0
        wx1 = x - x0
        wy0 = 1.0 - wy1
        wx0 = 1.0 - wx1
        w00 = (wy0 * wx0)
        w01 = (wy0 * wx1)
        w10 = (wy1 * wx0)
        w11 = (wy1 * wx1)
        w_vals = weight
        acc_w.index_put_((y0.long(), x0.long()), w_vals * w00, accumulate=True)
        acc_w.index_put_((y0.long(), x1.long()), w_vals * w01, accumulate=True)
        acc_w.index_put_((y1.long(), x0.long()), w_vals * w10, accumulate=True)
        acc_w.index_put_((y1.long(), x1.long()), w_vals * w11, accumulate=True)

    for _ in range(max_steps):
        if not alive.any():
            break

        alive_idx = torch.nonzero(alive, as_tuple=False).squeeze(1)
        cur_pos = pos[alive_idx]
        cur_dir = dir_vec[alive_idx]

        # Advance
        pos_new = cur_pos + cur_dir * step_size

        # Bounds check
        in_bounds = (
            (pos_new[:, 0] >= 0) & (pos_new[:, 0] < h - 1) &
            (pos_new[:, 1] >= 0) & (pos_new[:, 1] < w - 1)
        )

        live_idx = alive_idx[in_bounds]
        if live_idx.numel() == 0:
            alive[alive_idx] = False
            continue

        pos_new_ib = pos_new[in_bounds]

        # Refractive jump handling
        n_old = _bilinear_sample(n_t, cur_pos[in_bounds])
        n_new = _bilinear_sample(n_t, pos_new_ib)
        jump = (torch.abs(n_new - n_old) > 1e-4)
        if jump.any():
            normals = torch.stack([
                _bilinear_sample(ny, pos_new_ib[jump]),
                _bilinear_sample(nx, pos_new_ib[jump])
            ], dim=1)
            normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-6)
            new_dir, log_T = _snell_transmit(
                cur_dir[in_bounds][jump],
                n_old[jump],
                n_new[jump],
                normals,
            )
            cur_dir[in_bounds][jump] = new_dir
            # Transmission factor set to 1 (log_T=0) to avoid over-damping at interfaces

        # Update positions and directions
        pos[live_idx] = pos_new_ib
        dir_vec[live_idx] = cur_dir[in_bounds]

        # Integrate transport equation along the ray (intensity form)
        if integrate_transport:
            mid_pos = 0.5 * (cur_pos[in_bounds] + pos_new_ib)
            transport = _bilinear_sample(transport_field, mid_pos)
            weight[live_idx] *= torch.exp(-transport * step_size).clamp_min(1e-6)

        # Kill out-of-bounds
        dead = alive_idx[~in_bounds]
        alive[dead] = False

        # Rasterize (bilinear splat of intensity)
        y = pos_new_ib[:, 0].clamp(0, h - 1)
        x = pos_new_ib[:, 1].clamp(0, w - 1)
        y0 = torch.floor(y)
        x0 = torch.floor(x)
        y1 = torch.clamp(y0 + 1, max=h - 1)
        x1 = torch.clamp(x0 + 1, max=w - 1)
        wy1 = y - y0
        wx1 = x - x0
        wy0 = 1.0 - wy1
        wx0 = 1.0 - wx1
        w00 = (wy0 * wx0)
        w01 = (wy0 * wx1)
        w10 = (wy1 * wx0)
        w11 = (wy1 * wx1)
        w_vals = weight[live_idx]
        acc_w.index_put_((y0.long(), x0.long()), w_vals * w00, accumulate=True)
        acc_w.index_put_((y0.long(), x1.long()), w_vals * w01, accumulate=True)
        acc_w.index_put_((y1.long(), x0.long()), w_vals * w10, accumulate=True)
        acc_w.index_put_((y1.long(), x1.long()), w_vals * w11, accumulate=True)

    # Smooth density and derive amplitude
    acc_w_4d = acc_w.unsqueeze(0).unsqueeze(0)
    if blur_size <= 1:
        acc_blur = acc_w
    elif blur_size == 5 and blur_sigma is None:
        # Fast binomial 5x5 (default)
        g1d = torch.tensor([1., 4., 6., 4., 1.], device=device, dtype=torch.float32)
        g1d = g1d / g1d.sum()
        g2d = torch.outer(g1d, g1d)
        acc_blur = F.conv2d(acc_w_4d, g2d.view(1, 1, 5, 5), padding=2).squeeze(0).squeeze(0)
    else:
        # Generic Gaussian blur
        k = int(max(3, blur_size | 1))  # force odd, min 3
        if blur_sigma is None:
            sigma = k / 3.0
        else:
            sigma = float(blur_sigma)
        ax = torch.arange(k, device=device, dtype=torch.float32) - (k // 2)
        g1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = torch.outer(g1d, g1d)
        pad = k // 2
        acc_blur = F.conv2d(acc_w_4d, g2d.view(1, 1, k, k), padding=pad).squeeze(0).squeeze(0)

    density = torch.where(acc_blur > 1e-8, acc_blur, acc_w)
    A = torch.sqrt(torch.clamp(density, min=1e-8))

    # Normalize to 1 near source
    dist_out_t = torch.from_numpy(dist_out).to(device)
    src_band = (dist_out_t <= 2.5).to(device, dtype=torch.float32)
    if src_band.any():
        norm = (A * src_band).max().clamp_min(1e-3)
        A = A / norm
    A = torch.clamp(A, 0.0, 10.0)
    A[src_t > 0.5] = 1.0
    A_np = A.detach().cpu().numpy().astype(np.float32)

    if subsample <= 1:
        return A_np

    # Upsample to the original grid and re-normalize near the original source.
    A_full = _resample_to_shape(A_np, h_full, w_full)
    dist_out_full = distance_transform_edt((~source_mask).astype(np.uint8)).astype(np.float64)
    src_band_full = dist_out_full <= 2.5
    if src_band_full.any():
        norm = float(A_full[src_band_full].max())
        if norm > 1e-6:
            A_full = A_full / norm
    A_full[source_mask] = 1.0
    return A_full.astype(np.float32, copy=False)
