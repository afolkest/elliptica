"""
Eikonal equation PDE implementation (geometric optics).

Solves |∇φ|² = n(x,y)² for wavefront propagation using Fast Marching Method.
"""

import numpy as np
from typing import Any
from scipy.ndimage import zoom, distance_transform_edt, gaussian_filter
import skfmm
import os

from .base import PDEDefinition, BCField
from ..mask_utils import blur_mask
from .eikonal_amp import (
    trace_amplitude_torch,
    compute_amplitude_characteristic,
    compute_amplitude_characteristic_multires,
)

# Object type constants
SOURCE = 0
LENS = 1

# Edge BC types
EDGE_OPEN = 0      # Not a source, waves propagate through
EDGE_SOURCE = 1    # Plane wave source from this edge


def _compute_divergence_of_unit_gradient(phi: np.ndarray) -> np.ndarray:
    """Return ∇·v where v = ∇φ / |∇φ|."""
    gy, gx = np.gradient(phi)
    mag = np.hypot(gx, gy)
    mag = np.maximum(mag, 1e-6)
    vx = gx / mag
    vy = gy / mag
    div = np.gradient(vx, axis=1) + np.gradient(vy, axis=0)
    return div.astype(np.float64)


def compute_amplitude(phi: np.ndarray, n_field: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    """
    Solve the transport equation for amplitude: 2∇φ·∇A + A∇²φ = 0

    This gives intensity variations along rays - caustics appear as bright
    regions where rays converge (∇²φ < 0).

    Args:
        phi: Travel time / phase field from eikonal solve
        source_mask: Boolean mask of source pixels (A=1 there)

    Returns:
        Amplitude field A(x,y)
    """
    h, w = phi.shape
    phi64 = phi.astype(np.float64, copy=False)
    n64 = np.maximum(n_field.astype(np.float64, copy=False), 1e-6)

    # Smooth phi for more isotropic curvature estimates (FMM is first-order)
    phi_smooth = gaussian_filter(phi64, sigma=0.5, mode="nearest")

    # Direction field and curvature from smoothed phi
    gy, gx = np.gradient(phi_smooth)
    mag = np.hypot(gx, gy)
    mag = np.maximum(mag, 1e-6)
    sx = gx / mag
    sy = gy / mag
    div_s = np.gradient(sx, axis=1) + np.gradient(sy, axis=0)

    # Refractive index gradient contribution: s·∇(ln n)
    ln_n = np.log(n64)
    ln_n_y, ln_n_x = np.gradient(ln_n)
    dir_grad_ln_n = ln_n_x * sx + ln_n_y * sy

    # Combined term from ∇·(n s) = s·∇n + n∇·s -> (dir_grad_ln_n + div_s) * n
    # Transport along rays: d ln A / ds = -0.5 * (dir_grad_ln_n + div_s)
    transport_term = dir_grad_ln_n + div_s

    logA = np.full((h, w), np.inf, dtype=np.float64)
    logA[source_mask] = 0.0
    processed = source_mask.astype(bool, copy=True)

    order = np.argsort(phi64.ravel()).astype(np.int64)
    order_shape_w = w

    for idx in order:
        i = idx // order_shape_w
        j = idx % order_shape_w
        if processed[i, j]:
            continue

        best_phi = np.inf
        best_logA = 0.0

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = i + di
                nj = j + dj
                if ni < 0 or nj < 0 or ni >= h or nj >= w:
                    continue
                if not processed[ni, nj]:
                    continue

                phi_nb = phi64[ni, nj]
                if phi_nb < best_phi:
                    dphi = phi64[i, j] - phi_nb
                    n_avg = max(0.5 * (n64[i, j] + n64[ni, nj]), 1e-6)
                    ds = dphi / n_avg if dphi > 0 else 0.0
                    tau = 0.5 * (transport_term[i, j] + transport_term[ni, nj])
                    best_logA = logA[ni, nj] - tau * ds
                    best_phi = phi_nb

        if best_phi < np.inf:
            logA[i, j] = best_logA
        else:
            logA[i, j] = 0.0

        processed[i, j] = True

    logA = np.clip(logA, -30.0, 30.0)
    A = np.exp(logA)

    return A.astype(np.float32, copy=False)


def solve_eikonal(project: Any) -> dict[str, np.ndarray]:
    """
    Solve the eikonal equation for optical wavefront propagation.

    |∇φ| = n(x,y)  (or equivalently, |∇φ| = 1/speed)

    Uses Fast Marching Method via scikit-fmm.

    Objects can be:
    - Source: light origin (φ = 0 here)
    - Lens: region with refractive index n (high n ≈ blocker)

    Args:
        project: Project object with boundary_objects and shape

    Returns:
        Dictionary with 'phi' (travel time/phase) and 'n_field'
    """
    boundary_objects = project.boundary_objects
    grid_h, grid_w = project.shape

    # Initialize refractive index field (n=1 is free space)
    n_field = np.ones((grid_h, grid_w), dtype=np.float64)

    # Track source regions for LIC masking (sources block, lenses don't)
    source_mask = np.zeros((grid_h, grid_w), dtype=bool)

    has_source = False

    # Get domain scaling
    if hasattr(project, 'domain_size'):
        domain_w, domain_h = project.domain_size
        grid_scale_x = grid_w / domain_w if domain_w > 0 else 1.0
        grid_scale_y = grid_h / domain_h if domain_h > 0 else 1.0
    else:
        grid_scale_x = 1.0
        grid_scale_y = 1.0

    margin_x, margin_y = project.margin if hasattr(project, 'margin') else (0, 0)

    # No padding by default (UI already handles margins).
    pad = 0

    for obj in boundary_objects:
        # Get position with margin adjustment
        if hasattr(obj, 'position'):
            x = (obj.position[0] + margin_x) * grid_scale_x
            y = (obj.position[1] + margin_y) * grid_scale_y
        else:
            x = margin_x * grid_scale_x
            y = margin_y * grid_scale_y

        # Scale mask if needed
        obj_mask = obj.mask
        if not np.isclose(grid_scale_x, 1.0) or not np.isclose(grid_scale_y, 1.0):
            obj_mask = zoom(obj_mask, (grid_scale_y, grid_scale_x), order=0)

        # Apply edge smoothing if specified
        if hasattr(obj, 'edge_smooth_sigma') and obj.edge_smooth_sigma > 0:
            scale_factor = (grid_scale_x + grid_scale_y) / 2.0
            scaled_sigma = obj.edge_smooth_sigma * scale_factor
            obj_mask = blur_mask(obj_mask, scaled_sigma)

        # Place mask in grid
        mask_h, mask_w = obj_mask.shape
        ix, iy = int(round(x)), int(round(y))
        x0, y0 = max(0, ix), max(0, iy)
        x1, y1 = min(ix + mask_w, grid_w), min(iy + mask_h, grid_h)

        mx0, my0 = max(0, -ix), max(0, -iy)
        mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)

        if x1 <= x0 or y1 <= y0:
            continue

        mask_slice = obj_mask[my0:my1, mx0:mx1]
        mask_bool = mask_slice > 0.5

        # Get object type
        obj_type = obj.params.get('object_type', LENS)

        if obj_type == SOURCE:
            # Source: mark as negative in phi_init (wavefront origin)
            # Track as source region
            source_mask[y0:y1, x0:x1] |= mask_bool
            has_source = True
        else:
            # Lens: set refractive index
            n_value = obj.params.get('refractive_index', 1.5)
            n_field[y0:y1, x0:x1] = np.where(mask_bool, n_value, n_field[y0:y1, x0:x1])

    # Check edge boundary conditions for sources
    bc = project.boundary_conditions if hasattr(project, 'boundary_conditions') else {}
    # Also check the richer bc dict if available (has 'type' subfield)
    bc_rich = getattr(project, 'bc', {}) or {}

    def edge_is_source(edge_name):
        # Check rich BC first
        if edge_name in bc_rich:
            entry = bc_rich[edge_name]
            if isinstance(entry, dict):
                return entry.get('type', EDGE_OPEN) == EDGE_SOURCE
        # Fall back to simple BC
        return bc.get(edge_name, EDGE_OPEN) == EDGE_SOURCE

    if edge_is_source('left'):
        source_mask[:, 0] = True
        has_source = True
    if edge_is_source('right'):
        source_mask[:, -1] = True
        has_source = True
    if edge_is_source('top'):
        source_mask[0, :] = True
        has_source = True
    if edge_is_source('bottom'):
        source_mask[-1, :] = True
        has_source = True

    # Fallback: if still no sources, use left edge
    if not has_source:
        source_mask[:, 0] = True

    # Pad fields to reduce boundary bias (more free space for rays to exit)
    if pad > 0:
        n_field_p = np.pad(n_field, pad, mode="edge")
        source_mask_p = np.pad(source_mask, pad, constant_values=False)
    else:
        n_field_p = n_field
        source_mask_p = source_mask

    # Build signed-distance phi_init (zero at source boundary) for stable FMM
    outside_dist = distance_transform_edt(~source_mask_p)
    inside_dist = distance_transform_edt(source_mask_p)
    phi_init = outside_dist.astype(np.float64)
    phi_init[source_mask_p] = -inside_dist[source_mask_p]

    # Speed = 1/n (FMM uses speed, light slows in high-n media)
    speed = 1.0 / n_field_p

    # Solve eikonal with FMM (use second-order scheme for better isotropy)
    phi = skfmm.travel_time(phi_init, speed, order=2)

    # Handle any masked/invalid values
    if hasattr(phi, 'mask'):
        phi = np.where(phi.mask, 0.0, phi.data)

    # Enforce phi=0 inside sources for symmetry/stability of gradients
    phi = phi.astype(np.float64, copy=False)
    phi[source_mask_p] = 0.0

    # Solve transport equation for amplitude (caustics, intensity variations).
    # Defaults:
    #   - deterministic characteristic transport solve on the full grid;
    # Optional:
    #   - run characteristic on a restricted ROI around sources / lenses;
    #   - in homogeneous media only, run characteristic on a coarser grid and
    #     upsample the result back (multi-resolution).

    # Estimate background refractive index and lens regions.
    non_src = ~source_mask_p
    if non_src.any():
        n_bg = float(np.median(n_field_p[non_src]))
    else:
        n_bg = float(np.median(n_field_p))
    lens_mask = np.abs(n_field_p - n_bg) > 1e-4

    # Optional ROI: restrict the characteristic solve to a bounding box
    # around sources and lenses, with a configurable pixel margin.
    roi_env = os.environ.get("ELLIPTICA_EIKONAL_AMP_ROI", "0") == "1"
    roi_margin_env = os.environ.get("ELLIPTICA_EIKONAL_AMP_ROI_MARGIN", "64")
    try:
        roi_margin = max(0, int(roi_margin_env))
    except ValueError:
        roi_margin = 0

    phi_local = phi
    n_local = n_field_p
    source_local = source_mask_p
    roi_bbox = None

    if roi_env:
        roi_core = source_mask_p | lens_mask
        if roi_core.any():
            if roi_margin > 0:
                dist_roi = distance_transform_edt(~roi_core)
                roi_mask = dist_roi <= roi_margin
            else:
                roi_mask = roi_core

            rows = np.where(roi_mask.any(axis=1))[0]
            cols = np.where(roi_mask.any(axis=0))[0]
            if rows.size > 0 and cols.size > 0:
                y0, y1 = int(rows[0]), int(rows[-1]) + 1
                x0, x1 = int(cols[0]), int(cols[-1]) + 1
                # Only crop if ROI is strictly smaller than the full grid.
                if y0 > 0 or x0 > 0 or y1 < phi.shape[0] or x1 < phi.shape[1]:
                    phi_local = phi[y0:y1, x0:x1]
                    n_local = n_field_p[y0:y1, x0:x1]
                    source_local = source_mask_p[y0:y1, x0:x1]
                    roi_bbox = (y0, y1, x0, x1)

    amplitude_local: np.ndarray | None = None

    # Optional torch ray tracer remains opt-in via env flag.
    use_torch_amp = os.environ.get("ELLIPTICA_TORCH_EIKONAL_AMP", "1") != "0"
    torch_subsample_env = os.environ.get("ELLIPTICA_TORCH_AMP_SUBSAMPLE", "1")
    try:
        torch_subsample = max(1, int(torch_subsample_env))
    except ValueError:
        torch_subsample = 1

    if use_torch_amp:
        try:
            import torch  # Local import to avoid hard dependency during import time

            # Device selection: honor override, else CUDA > CPU (skip MPS by default;
            # MPS is slow for scatter-heavy splats in this tracer and is disabled).
            device_env = os.environ.get("ELLIPTICA_TORCH_DEVICE")
            if device_env:
                if device_env.lower() == "mps":
                    device = "cpu"
                else:
                    device = device_env
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            amplitude_local = trace_amplitude_torch(
                phi_local,
                n_local,
                source_local,
                step_size=0.75,
                seed_stride=1,
                jitter=0.2,
                max_steps=None,
                device=device,
                subsample=torch_subsample,
                target_rays=8000,
                integrate_transport=False,
            )
            msg = f"eikonal amplitude: using torch ray tracer on {device}"
            if torch_subsample > 1:
                msg += f" (subsample {torch_subsample}x)"
            if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                msg += " (MPS disabled for this solver)"
            print(msg, flush=True)
        except Exception:
            amplitude_local = None
            print("eikonal amplitude: torch path failed, falling back to deterministic characteristic solve", flush=True)

    if amplitude_local is None:
        # Optional multi-resolution characteristic solve. We only enable this when
        # the medium is homogeneous (no lens regions detected), since downsampling
        # has been observed to destroy caustic structure in variable-n fields.
        subsample_env = os.environ.get("ELLIPTICA_EIKONAL_AMP_SUBSAMPLE", "1")
        try:
            subsample = max(1, int(subsample_env))
        except ValueError:
            subsample = 1

        if subsample > 1 and not lens_mask.any():
            amplitude_local = compute_amplitude_characteristic_multires(
                phi_local,
                n_local,
                source_local,
                downsample=subsample,
                step_size=0.5,
            )
        else:
            amplitude_local = compute_amplitude_characteristic(
                phi_local,
                n_local,
                source_local,
                step_size=0.5,
            )

    if roi_bbox is not None:
        # Embed ROI amplitude into a full-resolution field and use A=1
        # outside the ROI as a cheap far-field approximation.
        amplitude = np.ones_like(phi, dtype=np.float32)
        y0, y1, x0, x1 = roi_bbox
        amplitude[y0:y1, x0:x1] = amplitude_local.astype(np.float32, copy=False)
        amplitude[source_mask_p] = 1.0
    else:
        amplitude = amplitude_local.astype(np.float32, copy=False)

    # Crop padding back to the original domain
    if pad > 0:
        sl = np.s_[pad:-pad, pad:-pad]
        phi = phi[sl]
        amplitude = amplitude[sl]

    # Sources block LIC (solid emitters), lenses don't (transparent)
    return {
        "phi": phi.astype(np.float32),
        "n_field": n_field.astype(np.float32),
        "amplitude": amplitude,
        "dirichlet_mask": source_mask,
    }


def extract_rays(solution: dict[str, np.ndarray], project: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract ray direction field from travel time.

    Rays propagate in direction of ∇φ (away from sources, perpendicular to wavefronts).

    Args:
        solution: Dictionary containing 'phi' (travel time)
        project: Project object (unused)

    Returns:
        Tuple of (ex, ey) ray direction components (normalized to unit vectors)
    """
    phi = solution["phi"]

    # Compute gradient (gy, gx order because numpy uses row-major)
    gy, gx = np.gradient(phi)

    # Normalize to unit vectors for LIC
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.where(mag < 1e-10, 1.0, mag)  # Avoid division by zero

    return gx / mag, gy / mag


# PDE Definition
EIKONAL_PDE = PDEDefinition(
    name="eikonal",
    display_name="Geometric Optics",
    description="Solve eikonal equation for light propagation (ray optics with refraction)",
    solve=solve_eikonal,
    extract_lic_field=extract_rays,
    boundary_params=[],
    boundary_fields=[
        BCField(
            name="object_type",
            display_name="Type",
            field_type="enum",
            default=LENS,
            choices=[("Source", SOURCE), ("Lens", LENS)],
            description="Source = light origin, Lens = refractive region"
        ),
        BCField(
            name="refractive_index",
            display_name="Refractive Index (n)",
            field_type="float",
            default=1.5,
            min_value=0.1,
            max_value=20.0,
            description="n>1 bends rays inward (converging), n<1 bends outward, high n ≈ blocker",
            visible_when={"object_type": LENS},
        ),
    ],
    bc_fields=[
        BCField(
            name="type",
            display_name="Edge Type",
            field_type="enum",
            default=EDGE_OPEN,
            choices=[("Open", EDGE_OPEN), ("Source (plane wave)", EDGE_SOURCE)],
            description="Open = waves pass through, Source = plane wave enters from this edge"
        ),
    ],
)
