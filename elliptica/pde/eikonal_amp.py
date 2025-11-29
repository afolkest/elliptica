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
import os
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation


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


def compute_amplitude_phi_ordered(
    phi: np.ndarray,
    n_field: np.ndarray,
    source_mask: np.ndarray,
    smooth_sigma: float = 0.5,
) -> np.ndarray:
    """
    Front-propagating (φ-ordered) transport solve for amplitude.

    This integrates d ln A / ds = -0.5 (∇·s + s·∇ln n) along characteristics
    s = ∇φ / |∇φ| using a single pass over the grid in increasing φ.

    We approximate the characteristic through each cell by a short segment
    connecting it to an upwind neighbour whose φ is smaller and whose
    direction is best aligned with the local ray direction.
    """
    h, w = phi.shape
    phi64 = phi.astype(np.float64, copy=False)
    n64 = np.maximum(n_field.astype(np.float64, copy=False), 1e-6)

    # Smooth phi to stabilise gradients and curvature (FMM is first order).
    if smooth_sigma > 0.0:
        phi_s = gaussian_filter(phi64, sigma=smooth_sigma, mode="nearest")
    else:
        phi_s = phi64

    gy, gx = np.gradient(phi_s)
    mag = np.hypot(gx, gy)
    mag = np.maximum(mag, 1e-8)
    sx = gx / mag
    sy = gy / mag

    # Divergence of the direction field.
    div_s = np.gradient(sx, axis=1) + np.gradient(sy, axis=0)

    # Refractive index contribution s·∇(ln n).
    ln_n = np.log(n64)
    ln_n_y, ln_n_x = np.gradient(ln_n)
    dir_grad_ln_n = ln_n_x * sx + ln_n_y * sy

    # Local transport term tau = div_s + s·∇ln n.
    tau = div_s + dir_grad_ln_n

    # Debias tau along wavefronts in the *background* medium only. For a
    # homogeneous region (n ≈ const), the correct transport is d ln A / ds = 0,
    # but FMM gradients introduce a small systematic bias in tau. We estimate
    # the mean tau on level-set bands of phi where n is close to the minimum
    # (background) value and subtract it there. Regions with significantly
    # different n (lenses) are left untouched so that true focusing is not
    # cancelled.
    n_min = float(n64.min())
    # Treat cells within a small epsilon of the background index as debias
    # candidates. This assumes piecewise-constant n with background at n_min.
    n_eps = 1e-3
    mask_bg = np.abs(n64 - n_min) < n_eps
    mask_ext = (~source_mask) & np.isfinite(phi64) & mask_bg
    if mask_ext.any():
        phi_vals = phi64[mask_ext]
        tau_vals = tau[mask_ext]
        phi_min = float(phi_vals.min())
        phi_max = float(phi_vals.max())
        if phi_max > phi_min:
            nbins = 64
            scale = (nbins - 1) / (phi_max - phi_min + 1e-12)
            idx = ((phi_vals - phi_min) * scale).astype(np.int64)
            idx = np.clip(idx, 0, nbins - 1)
            sums = np.zeros(nbins, dtype=np.float64)
            counts = np.zeros(nbins, dtype=np.int64)
            for k, tval in zip(idx, tau_vals):
                sums[k] += tval
                counts[k] += 1
            means = np.zeros(nbins, dtype=np.float64)
            nonzero = counts > 0
            means[nonzero] = sums[nonzero] / counts[nonzero]
            # Map band means back to full grid.
            phi_all = phi64[mask_ext]
            idx_all = ((phi_all - phi_min) * scale).astype(np.int64)
            idx_all = np.clip(idx_all, 0, nbins - 1)
            tau_debias = means[idx_all]
            tau_corr = tau.copy()
            tau_corr[mask_ext] = tau[mask_ext] - tau_debias
            tau = tau_corr

    # Initial conditions: logA = 0 on sources, +inf elsewhere.
    logA = np.full((h, w), np.inf, dtype=np.float64)
    logA[source_mask] = 0.0
    processed = source_mask.astype(bool, copy=True)

    # Process pixels in order of increasing phi (front propagation).
    order = np.argsort(phi64.ravel()).astype(np.int64)

    # Precompute neighbour offsets (8-connectivity).
    neighbours = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for idx in order:
        i = int(idx // w)
        j = int(idx % w)
        if processed[i, j]:
            continue

        phi_ij = phi64[i, j]
        if not np.isfinite(phi_ij):
            # Outside domain of definition for φ; keep A=1 for safety.
            logA[i, j] = 0.0
            processed[i, j] = True
            continue

        # Local ray direction.
        sx_ij = sx[i, j]
        sy_ij = sy[i, j]

        # Accumulate contributions from all upwind neighbours with positive
        # alignment to the local ray direction.
        w_sum = 0.0
        logA_acc = 0.0

        for di, dj in neighbours:
            ni = i + di
            nj = j + dj
            if ni < 0 or nj < 0 or ni >= h or nj >= w:
                continue
            if not processed[ni, nj]:
                continue

            phi_nb = phi64[ni, nj]
            if not np.isfinite(phi_nb):
                continue
            if phi_nb >= phi_ij:
                # Not upwind in φ.
                continue

            # Vector from neighbour to current cell (approximate ray step).
            dy = float(i - ni)
            dx = float(j - nj)
            r = (dx * dx + dy * dy) ** 0.5
            if r <= 0.0:
                continue
            d_hat_x = dx / r
            d_hat_y = dy / r

            # Alignment with local ray direction s.
            align = sx_ij * d_hat_x + sy_ij * d_hat_y
            if align <= 0.0:
                # Neighbour not upwind along the ray.
                continue

            # Prefer neighbours that are both well aligned and close in φ.
            dphi = phi_ij - phi_nb
            if dphi <= 1e-8:
                continue

            # Approximate arclength step along the ray.
            n_avg = max(0.5 * (n64[i, j] + n64[ni, nj]), 1e-6)
            ds = dphi / n_avg

            tau_avg = 0.5 * (tau[i, j] + tau[ni, nj])
            logA_nb = logA[ni, nj]
            logA_step = logA_nb - 0.5 * tau_avg * ds

            # Weight: alignment divided by φ gap.
            weight = align / dphi
            if weight > 0.0:
                w_sum += weight
                logA_acc += weight * logA_step

        if w_sum > 0.0:
            logA[i, j] = logA_acc / w_sum
        else:
            # Fallback: no well-aligned upwind neighbour found.
            # Keep A=1 here rather than injecting spurious damping.
            logA[i, j] = 0.0

        processed[i, j] = True

    logA = np.clip(logA, -30.0, 30.0)
    A = np.exp(logA)

    # Normalize near the source to keep A≈1 in a small band, mirroring the
    # characteristic solver and removing residual global drift.
    dist_out = distance_transform_edt((~source_mask).astype(np.uint8)).astype(np.float64)
    src_band = dist_out <= 2.5
    if src_band.any():
        norm = float(A[src_band].max())
        if norm > 1e-6:
            A = A / norm

    A[source_mask] = 1.0
    return A.astype(np.float32, copy=False)


def compute_amplitude_characteristic(
    phi: np.ndarray,
    n_field: np.ndarray,
    source_mask: np.ndarray,
    step_size: float = 0.5,
) -> np.ndarray:
    """
    Deterministic characteristic transport solve for amplitude:
    s·∇A = -0.5 A (∇·s + s·∇ln n), where s = ∇φ / |∇φ|.

    Backtrack along characteristics using a semi-Lagrangian step on the
    continuous direction field s, one ray per grid point.
    """
    h, w = phi.shape
    phi64 = phi.astype(np.float64, copy=False)
    n64 = np.maximum(n_field.astype(np.float64, copy=False), 1e-6)

    # Smooth phi to stabilize gradients
    phi_s = gaussian_filter(phi64, sigma=0.5, mode="nearest")

    gy, gx = np.gradient(phi_s)
    mag = np.hypot(gx, gy)
    mag = np.maximum(mag, 1e-8)
    sx = gx / mag
    sy = gy / mag
    div_s = np.gradient(sx, axis=1) + np.gradient(sy, axis=0)

    ln_n = np.log(n64)
    ln_n_y, ln_n_x = np.gradient(ln_n)
    transport = div_s + ln_n_x * sx + ln_n_y * sy
    transport = np.clip(transport, -10.0, 10.0)

    # Distance to source (for stopping criterion)
    dist_src = distance_transform_edt((~source_mask).astype(np.uint8)).astype(np.float64)

    # Initialize backtracking positions and log-amplitude
    Y, X = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    Ycur = Y.copy()
    Xcur = X.copy()

    logA = np.zeros((h, w), dtype=np.float64)
    active = ~source_mask.copy()

    # Conservative upper bound on steps (domain diameter in pixels / step_size)
    max_steps = int((dist_src.max() + 2.0) / max(step_size, 1e-3))

    for _ in range(max_steps):
        if not active.any():
            break

        y_flat = Ycur.ravel()
        x_flat = Xcur.ravel()

        sy_flat = _bilinear_sample_np(sy, y_flat, x_flat).reshape(h, w)
        sx_flat = _bilinear_sample_np(sx, y_flat, x_flat).reshape(h, w)
        t_flat = _bilinear_sample_np(transport, y_flat, x_flat).reshape(h, w)

        sy_step = sy_flat * active
        sx_step = sx_flat * active
        t_step = t_flat * active

        Ycur = np.clip(Ycur - sy_step * step_size, 0.0, h - 1.0)
        Xcur = np.clip(Xcur - sx_step * step_size, 0.0, w - 1.0)

        logA -= 0.5 * t_step * step_size

        # Check which rays have reached the source band
        d_here = _bilinear_sample_np(dist_src, Ycur.ravel(), Xcur.ravel()).reshape(h, w)
        arrived = active & (d_here <= step_size)
        if arrived.any():
            logA[arrived] = 0.0  # enforce A=1 at the source
            active[arrived] = False

    # Normalize near source to 1
    dist_out = distance_transform_edt((~source_mask).astype(np.uint8)).astype(np.float64)
    src_band = dist_out <= 2.5
    if src_band.any():
        A = np.exp(logA)
        norm = np.max(A[src_band])
        if norm > 1e-6:
            A = A / norm
    else:
        A = np.exp(logA)
    A[source_mask] = 1.0

    return A.astype(np.float32, copy=False)


def compute_amplitude_characteristic_multires(
    phi: np.ndarray,
    n_field: np.ndarray,
    source_mask: np.ndarray,
    downsample: int = 1,
    step_size: float = 0.5,
) -> np.ndarray:
    """
    Multi-resolution wrapper around the characteristic transport solver.

    The characteristic solver is treated as ground truth but run on a
    downsampled grid for speed, then the resulting amplitude is upsampled
    back to the original resolution.

    Args:
        phi: Full-resolution eikonal field.
        n_field: Full-resolution refractive index field.
        source_mask: Full-resolution boolean source mask (A=1 inside).
        downsample: Integer downsampling factor (>=1). A value of 1 reduces
            to the plain characteristic solver.
        step_size: Step size in grid units for the *full* resolution. The
            effective step on the coarse grid is scaled by 1/downsample so
            that the physical step length is approximately preserved.
    """
    if downsample <= 1:
        return compute_amplitude_characteristic(phi, n_field, source_mask, step_size=step_size)

    phi = np.asarray(phi)
    n_field = np.asarray(n_field)
    source_mask = np.asarray(source_mask, dtype=bool)

    h, w = phi.shape
    ds = int(max(1, downsample))

    # Target coarse resolution; we cover the full domain by sampling on a
    # regular lattice of size (hc, wc).
    hc = max(1, int(np.ceil(h / ds)))
    wc = max(1, int(np.ceil(w / ds)))

    phi_coarse = _resample_to_shape(phi, hc, wc)
    n_coarse = _resample_to_shape(n_field, hc, wc)

    # For the source mask, resample the 0/1 field and threshold to preserve
    # connectivity. Any coarse cell that sees >0.5 coverage becomes a source.
    src_float = source_mask.astype(np.float64)
    src_coarse_f = _resample_to_shape(src_float, hc, wc)
    src_coarse = src_coarse_f > 0.5

    # Scale the step size so that the physical step length roughly matches
    # the full-resolution solver.
    step_coarse = float(step_size) / float(ds)

    A_coarse = compute_amplitude_characteristic(
        phi_coarse,
        n_coarse,
        src_coarse,
        step_size=step_coarse,
    ).astype(np.float64)

    # Upsample amplitude back to the original grid.
    A_full = _resample_to_shape(A_coarse, h, w)

    # Re-normalise near the original source to keep A≈1 in a small band,
    # mirroring the characteristic solver.
    dist_out = distance_transform_edt((~source_mask).astype(np.uint8)).astype(np.float64)
    src_band = dist_out <= 2.5
    if src_band.any():
        norm = float(A_full[src_band].max())
        if norm > 1e-6:
            A_full = A_full / norm
    A_full[source_mask] = 1.0

    return A_full.astype(np.float32, copy=False)


def compute_amplitude_characteristic_torch(
    phi: np.ndarray,
    n_field: np.ndarray,
    source_mask: np.ndarray,
    step_size: float = 0.75,
    max_steps: int | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """
    Torch implementation of characteristic transport solve (one ray per pixel).

    This mirrors compute_amplitude_characteristic but runs the backtracking
    loop on a torch device (CPU, CUDA, or MPS) for speed.
    """
    device = torch.device(device)
    if device.type == "mps":
        # Force CPU; MPS is slow for the scatter-heavy tracer.
        device = torch.device("cpu")
    h, w = phi.shape

    phi64 = phi.astype(np.float64, copy=False)
    n64 = np.maximum(n_field.astype(np.float64, copy=False), 1e-6)

    # Smooth phi to stabilize gradients
    phi_s = gaussian_filter(phi64, sigma=0.5, mode="nearest")

    gy, gx = np.gradient(phi_s)
    mag = np.hypot(gx, gy)
    mag = np.maximum(mag, 1e-8)
    sx_np = gx / mag
    sy_np = gy / mag
    div_s = np.gradient(sx_np, axis=1) + np.gradient(sy_np, axis=0)

    ln_n = np.log(n64)
    ln_n_y, ln_n_x = np.gradient(ln_n)
    transport_np = div_s + ln_n_x * sx_np + ln_n_y * sy_np
    transport_np = np.clip(transport_np, -10.0, 10.0)

    dist_src_np = distance_transform_edt((~source_mask).astype(np.uint8)).astype(np.float64)
    if max_steps is None:
        max_steps = int((dist_src_np.max() + 2.0) / max(step_size, 1e-3))

    sx = torch.from_numpy(sx_np.astype(np.float32)).to(device)
    sy = torch.from_numpy(sy_np.astype(np.float32)).to(device)
    transport = torch.from_numpy(transport_np.astype(np.float32)).to(device)
    dist_src = torch.from_numpy(dist_src_np.astype(np.float32)).to(device)
    src_mask_t = torch.from_numpy(source_mask.astype(np.bool_)).to(device)

    # Grid of starting positions
    Y = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1).expand(h, w)
    X = torch.arange(w, device=device, dtype=torch.float32).view(1, -1).expand(h, w)
    Ycur = Y.clone()
    Xcur = X.clone()

    logA = torch.zeros((h, w), device=device, dtype=torch.float32)
    active = (~src_mask_t).clone()

    for _ in range(max_steps):
        if not active.any():
            break

        coords = torch.stack([Ycur.reshape(-1), Xcur.reshape(-1)], dim=1)
        sy_flat = _bilinear_sample(sy, coords).reshape(h, w)
        sx_flat = _bilinear_sample(sx, coords).reshape(h, w)
        t_flat = _bilinear_sample(transport, coords).reshape(h, w)

        sy_step = sy_flat * active
        sx_step = sx_flat * active
        t_step = t_flat * active

        Ycur = (Ycur - sy_step * step_size).clamp(0.0, h - 1.0)
        Xcur = (Xcur - sx_step * step_size).clamp(0.0, w - 1.0)

        logA = logA - 0.5 * t_step * step_size

        # Check which rays have reached the source band
        coords2 = torch.stack([Ycur.reshape(-1), Xcur.reshape(-1)], dim=1)
        d_here = _bilinear_sample(dist_src, coords2).reshape(h, w)
        arrived = active & (d_here <= step_size)
        if arrived.any():
            logA[arrived] = 0.0
            active[arrived] = False

    A = torch.exp(logA)

    # Normalize near source to 1
    dist_out_np = distance_transform_edt((~source_mask).astype(np.uint8)).astype(np.float32)
    src_band = torch.from_numpy((dist_out_np <= 2.5).astype(np.bool_)).to(device)
    if src_band.any():
        norm = A[src_band].max().clamp_min(1e-6)
        A = A / norm
    A[src_mask_t] = 1.0

    return A.detach().cpu().numpy().astype(np.float32)


def compute_amplitude_flux(
    phi: np.ndarray,
    n_field: np.ndarray,
    source_mask: np.ndarray,
    n_iters: int = 60,
    cfl: float = 0.8,
    pad_fraction: float = 0.1,
    pad_min: int = 32,
    diffusion: float = 0.02,
) -> np.ndarray:
    """
    Grid-based transport solve for intensity Q = A^2 via a conservation form.

    This approximates geometric-optics amplitude variations (caustics,
    focusing) by evolving Q toward the steady state of ∇·(Q ∇φ) ≈ 0 with
    Dirichlet A=1 on the source set.

    This is O(N * n_iters) with each iteration fully vectorized.
    """
    # Optional padding to reduce influence of outer box on interior.
    h0, w0 = phi.shape

    # Base pad from domain size
    base_pad = int(max(pad_min, pad_fraction * max(h0, w0)))

    # Additional pad to ensure sources are not too close to the computational boundary
    y_idx, x_idx = np.nonzero(source_mask)
    if y_idx.size > 0:
        d_top = y_idx.min()
        d_bottom = h0 - 1 - y_idx.max()
        d_left = x_idx.min()
        d_right = w0 - 1 - x_idx.max()
        min_d = float(min(d_top, d_bottom, d_left, d_right))
        target_margin = float(base_pad)
        extra_pad = max(0.0, target_margin - min_d)
    else:
        extra_pad = 0.0

    pad = int(max(base_pad, extra_pad))
    if pad > 0:
        phi64 = np.pad(phi.astype(np.float64, copy=False), pad, mode="edge")
        n64 = np.pad(n_field.astype(np.float64, copy=False), pad, mode="edge")
        src_pad = np.pad(source_mask.astype(bool), pad, constant_values=False)
    else:
        phi64 = phi.astype(np.float64, copy=False)
        n64 = n_field.astype(np.float64, copy=False)
        src_pad = source_mask.astype(bool)

    # Slightly thicken the Dirichlet region for amplitude to reduce
    # grid-scale halos right at the discrete source boundary.
    bc_mask = binary_dilation(src_pad, iterations=2)

    # Velocity field v = ∇φ (not normalized).
    gy, gx = np.gradient(phi64)
    vx = gx
    vy = gy

    h, w = phi64.shape

    # Time step from CFL condition based on max speed.
    speed = np.abs(vx) + np.abs(vy)
    vmax = float(speed.max())
    if vmax < 1e-6:
        A = np.ones_like(phi, dtype=np.float32)
        A[source_mask] = 1.0
        return A
    dt = cfl / vmax

    # Intensity Q = A^2; start from 1 everywhere.
    Q = np.ones((h, w), dtype=np.float64)
    Q[bc_mask] = 1.0

    for _ in range(n_iters):
        # Face velocities in x (between columns j and j+1).
        vx_face = 0.5 * (vx[:, :-1] + vx[:, 1:])  # shape (h, w-1)
        Q_left = Q[:, :-1]
        Q_right = Q[:, 1:]
        upwind_x = np.where(vx_face >= 0.0, Q_left, Q_right)
        flux_x = vx_face * upwind_x  # flux through faces j+1/2, positive to the right

        # Divergence in x: F_{j+1/2} - F_{j-1/2}.
        div_x = np.zeros_like(Q)
        div_x[:, 1:-1] = flux_x[:, 1:] - flux_x[:, :-1]

        # Transparent-ish outflow boundaries in x:
        # - No inflow from outside (Q_out = 0 for upwind),
        # - Allow outflow using interior Q and local velocity.
        vx_left = vx[:, 0]
        vx_right = vx[:, -1]
        flux_left = np.where(vx_left < 0.0, vx_left * Q[:, 0], 0.0)
        flux_right = np.where(vx_right > 0.0, vx_right * Q[:, -1], 0.0)
        div_x[:, 0] = flux_x[:, 0] - flux_left
        div_x[:, -1] = flux_right - flux_x[:, -1]

        # Face velocities in y (between rows i and i+1).
        vy_face = 0.5 * (vy[:-1, :] + vy[1:, :])  # shape (h-1, w)
        Q_down = Q[:-1, :]
        Q_up = Q[1:, :]
        upwind_y = np.where(vy_face >= 0.0, Q_down, Q_up)
        flux_y = vy_face * upwind_y  # flux through faces i+1/2, positive upwards (in y)

        div_y = np.zeros_like(Q)
        div_y[1:-1, :] = flux_y[1:, :] - flux_y[:-1, :]

        # Same outflow-only treatment for y boundaries.
        vy_bottom = vy[0, :]
        vy_top = vy[-1, :]
        flux_bottom = np.where(vy_bottom < 0.0, vy_bottom * Q[0, :], 0.0)
        flux_top = np.where(vy_top > 0.0, vy_top * Q[-1, :], 0.0)
        div_y[0, :] = flux_y[0, :] - flux_bottom
        div_y[-1, :] = flux_top - flux_y[-1, :]

        div = div_x + div_y

        Q = Q - dt * div

        # Small isotropic diffusion step to reduce grid-aligned artefacts.
        if diffusion > 0.0:
            lap = np.zeros_like(Q)
            lap[1:-1, 1:-1] = (
                Q[1:-1, 0:-2] + Q[1:-1, 2:] +
                Q[0:-2, 1:-1] + Q[2:, 1:-1] -
                4.0 * Q[1:-1, 1:-1]
            )
            Q = Q + diffusion * lap

        Q = np.maximum(Q, 0.0)
        Q[bc_mask] = 1.0

    A = np.sqrt(np.maximum(Q, 1e-8))

    # Crop back to original domain if padded.
    if pad > 0:
        A = A[pad:pad + h0, pad:pad + w0]

    # Normalize to 1 near the source (for numerical drift) on the original domain.
    dist_out = distance_transform_edt((~source_mask).astype(np.uint8)).astype(np.float64)
    src_band = dist_out <= 2.5
    if src_band.any():
        norm = float(A[src_band].max())
        if norm > 1e-6:
            A = A / norm
    A[source_mask] = 1.0
    return A.astype(np.float32, copy=False)


def compute_amplitude_semi_lagrangian(
    phi: np.ndarray,
    n_field: np.ndarray,
    source_mask: np.ndarray,
    step_size: float = 0.75,
    n_iters: int = 40,
    smooth_sigma: float = 0.5,
    kappa_clip: float = 10.0,
    q_min: float = 1e-6,
    q_max: float = 1e6,
) -> np.ndarray:
    """
    Semi-Lagrangian fixed-point solve for intensity Q = A^2 along rays.

    This approximates the conservation law ∇·(Q ∇φ) = 0 by iterating a
    short backtrace along the ray direction field s = ∇φ / |∇φ|:

        ln(Q |∇φ|)(x) ≈ ln(Q |∇φ|)(x - Δℓ s) - Δℓ ∇·s(x)

    which yields the update

        Q_new(x) = (|∇φ|(x - Δℓ s) / |∇φ|(x)) * Q(x - Δℓ s) * exp(-Δℓ κ(x)),

    where κ = ∇·s. Boundary conditions are enforced by fixing Q=1 on the
    source mask. This is O(N * n_iters) with each iteration fully vectorised.
    """
    h, w = phi.shape
    phi64 = phi.astype(np.float64, copy=False)

    # Smooth phi to stabilise gradients (FMM is first order).
    if smooth_sigma > 0.0:
        phi_s = gaussian_filter(phi64, sigma=smooth_sigma, mode="nearest")
    else:
        phi_s = phi64

    gy, gx = np.gradient(phi_s)
    speed = np.hypot(gx, gy)
    speed = np.maximum(speed, 1e-8)

    sx = gx / speed
    sy = gy / speed

    # Divergence of the unit direction field.
    kappa = np.gradient(sx, axis=1) + np.gradient(sy, axis=0)
    if kappa_clip is not None:
        kappa = np.clip(kappa, -kappa_clip, kappa_clip)

    # Coordinate grid for backtracing.
    Y, X = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )

    # Initialise Q = A^2; start from 1 everywhere.
    Q = np.ones((h, w), dtype=np.float64)
    Q[source_mask] = 1.0

    for _ in range(n_iters):
        # Backtrace one step along -s from every grid point.
        Yp = Y - sy * step_size
        Xp = X - sx * step_size

        # Clamp to domain.
        Yp = np.clip(Yp, 0.0, h - 1.0)
        Xp = np.clip(Xp, 0.0, w - 1.0)

        # Sample Q and |∇φ| at backtraced positions.
        y_flat = Yp.ravel()
        x_flat = Xp.ravel()
        Q_up = _bilinear_sample_np(Q, y_flat, x_flat).reshape(h, w)
        speed_up = _bilinear_sample_np(speed, y_flat, x_flat).reshape(h, w)

        # Semi-Lagrangian update.
        ratio = speed_up / speed
        Q_new = ratio * Q_up * np.exp(-step_size * kappa)

        # Clamp and enforce boundary conditions.
        Q_new = np.clip(Q_new, q_min, q_max)
        Q_new[source_mask] = 1.0
        Q = Q_new

    A = np.sqrt(Q)

    # Normalize to 1 near the source to remove residual drift.
    dist_out = distance_transform_edt((~source_mask).astype(np.uint8)).astype(np.float64)
    src_band = dist_out <= 2.5
    if src_band.any():
        norm = float(A[src_band].max())
        if norm > 1e-6:
            A = A / norm
    A[source_mask] = 1.0
    return A.astype(np.float32, copy=False)


def compute_amplitude_jacobian(
    phi: np.ndarray,
    n_field: np.ndarray,
    source_mask: np.ndarray,
    step_size: float = 0.75,
    n_iters: int = 40,
    smooth_sigma: float = 0.5,
) -> np.ndarray:
    """
    Amplitude from ray-label Jacobian: A ∝ 1 / sqrt(|∇φ × ∇σ|).

    We solve the advection equation ∇φ·∇σ = 0 to extend a scalar ray label σ
    defined on the source region along characteristics, then compute a local
    geometric spreading factor from the Jacobian determinant of the mapping
    (source label, travel time) -> (x,y).

    The label σ is initialised on the source using the polar angle around the
    source centroid. This is exact for circular sources and a reasonable
    surrogate for more general shapes in simple tests.
    """
    h, w = phi.shape
    phi64 = phi.astype(np.float64, copy=False)

    # Smooth phi to stabilise gradients (FMM is first order).
    if smooth_sigma > 0.0:
        phi_s = gaussian_filter(phi64, sigma=smooth_sigma, mode="nearest")
    else:
        phi_s = phi64

    gy, gx = np.gradient(phi_s)
    speed = np.hypot(gx, gy)
    speed = np.maximum(speed, 1e-8)

    sx = gx / speed
    sy = gy / speed

    # Coordinate grid for backtracing.
    Y, X = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )

    # Initialise ray label σ on the source boundary: use polar angle around centroid.
    src_idx = np.nonzero(source_mask)
    if src_idx[0].size == 0:
        # No sources; return uniform amplitude.
        A = np.ones_like(phi, dtype=np.float32)
        return A

    cy = float(src_idx[0].mean())
    cx = float(src_idx[1].mean())
    dy = Y - cy
    dx = X - cx
    sigma = np.zeros((h, w), dtype=np.float64)

    # Use a 1-pixel thick boundary ring of the source as the Dirichlet set
    # for σ. Interior source pixels are not constrained so that the mapping
    # (σ, τ) -> (x, y) is controlled by the boundary geometry rather than
    # arbitrary interior values.
    from scipy.ndimage import binary_dilation

    dilated = binary_dilation(source_mask)
    src_ring = source_mask & (~(dilated & source_mask))
    # Fallback: if morphology fails (e.g. single pixel), treat whole source
    # region as the ring.
    if not src_ring.any():
        src_ring = source_mask.copy()

    sigma[src_ring] = np.arctan2(dy[src_ring], dx[src_ring])

    # Semi-Lagrangian advection: ∇φ·∇σ = 0 -> σ constant along characteristics.
    for _ in range(n_iters):
        # Backtrace one step along -s from every grid point.
        Yp = Y - sy * step_size
        Xp = X - sx * step_size

        Yp = np.clip(Yp, 0.0, h - 1.0)
        Xp = np.clip(Xp, 0.0, w - 1.0)

        y_flat = Yp.ravel()
        x_flat = Xp.ravel()
        sigma_up = _bilinear_sample_np(sigma, y_flat, x_flat).reshape(h, w)

        sigma_new = sigma_up

        # Enforce boundary condition on the source boundary ring only.
        sigma_new[src_ring] = sigma[src_ring]
        sigma = sigma_new

    # Compute gradients of φ and σ.
    phi_y, phi_x = np.gradient(phi_s)
    sigma_y, sigma_x = np.gradient(sigma)

    # Jacobian determinant J = |∇φ × ∇σ| (2D scalar cross product).
    J = phi_x * sigma_y - phi_y * sigma_x
    J_abs = np.abs(J)

    eps = 1e-6
    A = 1.0 / np.sqrt(J_abs + eps)

    # Normalize near the source and enforce A=1 on the source.
    dist_out = distance_transform_edt((~source_mask).astype(np.uint8)).astype(np.float64)
    src_band = dist_out <= 2.5
    if src_band.any():
        norm = float(A[src_band].max())
        if norm > 1e-6:
            A = A / norm
    A[source_mask] = 1.0
    return A.astype(np.float32, copy=False)


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
    step_size: float = 0.75,
    seed_stride: int = 1,
    jitter: float = 0.0,
    integrate_transport: bool = False,
    max_steps: int | None = None,
    device: str | torch.device = "cpu",
    subsample: int = 1,
    target_rays: int = 4000,
    blur_size: int = 5,
    blur_sigma: float | None = None,
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
    # Env overrides for tuning without code changes.
    step_size = float(os.environ.get("ELLIPTICA_TORCH_AMP_STEP", step_size))
    jitter = float(os.environ.get("ELLIPTICA_TORCH_AMP_JITTER", jitter))
    integrate_transport = os.environ.get("ELLIPTICA_TORCH_AMP_TRANSPORT", "0") == "1" if "ELLIPTICA_TORCH_AMP_TRANSPORT" in os.environ else integrate_transport
    target_rays = int(os.environ.get("ELLIPTICA_TORCH_AMP_TARGET_RAYS", target_rays))
    blur_size = int(os.environ.get("ELLIPTICA_TORCH_AMP_BLUR_SIZE", blur_size))
    if "ELLIPTICA_TORCH_AMP_BLUR_SIGMA" in os.environ:
        blur_sigma = float(os.environ["ELLIPTICA_TORCH_AMP_BLUR_SIGMA"])
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
