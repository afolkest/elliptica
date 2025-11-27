"""
Biharmonic equation PDE implementation (Stokes flow / clamped plate).
Mixed formulation: solve Δw = 0 for w = Δφ, then Δφ = w with Dirichlet BCs.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict
from scipy.sparse import coo_matrix
from pyamg import smoothed_aggregation_solver
from ..poisson import DIRICHLET, NEUMANN, solve_poisson_system
from .base import PDEDefinition, BoundaryParameter, BCField
from .poisson_pde import extract_electric_field
from .boundary_utils import build_dirichlet_from_objects
from scipy.ndimage import convolve, zoom
from scipy.sparse.linalg import cg
from ..mask_utils import blur_mask

# Interior boundary condition types for biharmonic
BIHARMONIC_CLAMPED = 0          # φ = V, ∂ₙφ ≈ 0 (default, uses band extension)
BIHARMONIC_SIMPLY_SUPPORTED = 1  # φ = V, ∇²φ = g (specify Laplacian)
BIHARMONIC_FLUX = 2              # φ = V, ∂ₙφ = g (specify normal flux)


def _extend_dirichlet_band(mask: np.ndarray, values: np.ndarray, steps: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Grow a constant-value Dirichlet band to approximate zero normal slope."""
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.uint8)
    mask_out = mask.copy()
    values_out = values.copy()

    for _ in range(steps):
        neighbor_counts = convolve(mask_out.astype(np.uint8), kernel, mode="constant", cval=0)
        band = (neighbor_counts > 0) & (~mask_out)
        if not band.any():
            break
        weighted = convolve(values_out * mask_out, kernel, mode="constant", cval=0.0)
        nz = neighbor_counts.astype(float)
        nz[nz == 0] = 1.0
        band_values = weighted / nz
        values_out = values_out.copy()
        values_out[band] = band_values[band]
        mask_out = mask_out | band

    return mask_out, values_out


def _edge_bc_from_map(bc_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Normalize boundary condition entries from the project.bc map."""
    edges: Dict[str, Dict[str, float]] = {}
    for edge in ("top", "bottom", "left", "right"):
        entry = bc_map.get(edge, {}) or {}
        edges[edge] = {
            "type": int(entry.get("type", DIRICHLET)),
            # value/slope/laplacian may be scalars or 1D arrays (variable along the edge)
            "value": entry.get("value", 0.0),
            "laplacian": entry.get("laplacian", 0.0),
            "slope": entry.get("slope", 0.0),
        }
    return edges


def _edge_array(val: Any, length: int) -> np.ndarray:
    """Coerce scalar-or-array to a 1D array of a given length."""
    arr = np.asarray(val)
    if arr.shape == ():  # scalar
        return np.full(length, float(arr), dtype=float)
    if arr.size != length:
        arr = np.resize(arr, length)
    return arr.astype(float, copy=False)


def _edge_sample(val: Any, idx: int, length: int) -> float:
    """Sample scalar-or-array edge quantity at a tangential index."""
    arr = np.asarray(val)
    if arr.shape == ():
        return float(arr)
    if arr.size == 0:
        return 0.0
    if arr.size != length:
        arr = np.resize(arr, length)
    idx = max(0, min(length - 1, idx))
    return float(arr[idx])


def _apply_edge_dirichlet(mask: np.ndarray, values: np.ndarray, edge_bc: Dict[str, Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Inject edge Dirichlet conditions directly into the mask/values arrays."""
    mask_out = mask.copy()
    values_out = values.copy()
    h, w = mask.shape

    # Edge rows/cols
    if edge_bc["top"]["type"] == DIRICHLET:
        row_mask = mask_out[0, :]
        top_vals = _edge_array(edge_bc["top"]["value"], w)
        values_out[0, :] = np.where(row_mask, values_out[0, :], top_vals)
        mask_out[0, :] = True
    if edge_bc["bottom"]["type"] == DIRICHLET:
        row_mask = mask_out[h - 1, :]
        bottom_vals = _edge_array(edge_bc["bottom"]["value"], w)
        values_out[h - 1, :] = np.where(row_mask, values_out[h - 1, :], bottom_vals)
        mask_out[h - 1, :] = True
    if edge_bc["left"]["type"] == DIRICHLET:
        col_mask = mask_out[:, 0]
        left_vals = _edge_array(edge_bc["left"]["value"], h)
        values_out[:, 0] = np.where(col_mask, values_out[:, 0], left_vals)
        mask_out[:, 0] = True
    if edge_bc["right"]["type"] == DIRICHLET:
        col_mask = mask_out[:, w - 1]
        right_vals = _edge_array(edge_bc["right"]["value"], h)
        values_out[:, w - 1] = np.where(col_mask, values_out[:, w - 1], right_vals)
        mask_out[:, w - 1] = True

    # Corners: average participating edge values if not already set by an object
    corners = [
        ((0, 0), ("top", "left")),
        ((0, w - 1), ("top", "right")),
        ((h - 1, 0), ("bottom", "left")),
        ((h - 1, w - 1), ("bottom", "right")),
    ]
    for (ci, cj), edge_names in corners:
        if mask_out[ci, cj]:
            continue
        vals = []
        for e in edge_names:
            if edge_bc[e]["type"] != DIRICHLET:
                continue
            if e in ("top", "bottom"):
                vals.append(_edge_sample(edge_bc[e]["value"], cj, w))
            else:
                vals.append(_edge_sample(edge_bc[e]["value"], ci, h))
        if vals:
            mask_out[ci, cj] = True
            values_out[ci, cj] = float(np.mean(vals))

    return mask_out, values_out


def _solve_poisson_with_interior_neumann(
    dirichlet_mask: np.ndarray,
    dirichlet_values: np.ndarray,
    neumann_mask: np.ndarray,
    neumann_values: np.ndarray,
    edge_bc: Dict[str, Dict[str, Any]],
    hx: float,
    hy: float,
    rhs_source: np.ndarray,
) -> np.ndarray:
    """Solve Δφ = rhs_source with Dirichlet + interior Neumann boundaries.

    This extends _solve_poisson_with_flux to handle interior objects with
    Neumann BCs (∂ₙφ = g) using ghost cell approach.

    Args:
        dirichlet_mask: Boolean mask for Dirichlet pixels (φ = V)
        dirichlet_values: Values at Dirichlet pixels
        neumann_mask: Boolean mask for interior Neumann pixels
        neumann_values: Flux values (∂ₙφ) at Neumann pixels
        edge_bc: Domain edge boundary conditions
        hx, hy: Grid spacing
        rhs_source: Source term (w field from first biharmonic solve)

    Returns:
        Solution φ field
    """
    h, w = dirichlet_mask.shape
    N = h * w
    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    rhs = -rhs_source.astype(np.float64).ravel()

    def idx(i: int, j: int) -> int:
        return j + i * w

    top_vals = _edge_array(edge_bc["top"]["value"], w)
    bottom_vals = _edge_array(edge_bc["bottom"]["value"], w)
    left_vals = _edge_array(edge_bc["left"]["value"], h)
    right_vals = _edge_array(edge_bc["right"]["value"], h)

    top_slope = _edge_array(edge_bc["top"].get("slope", 0.0), w)
    bottom_slope = _edge_array(edge_bc["bottom"].get("slope", 0.0), w)
    left_slope = _edge_array(edge_bc["left"].get("slope", 0.0), h)
    right_slope = _edge_array(edge_bc["right"].get("slope", 0.0), h)

    for i in range(h):
        for j in range(w):
            k = idx(i, j)

            # Check if this pixel is in a Neumann region (interior object with flux BC)
            in_neumann = neumann_mask[i, j]

            if dirichlet_mask[i, j]:
                # Dirichlet pixel: φ = V
                row_idx.append(k)
                col_idx.append(k)
                data.append(1.0)
                rhs[k] = dirichlet_values[i, j]
                continue

            if in_neumann:
                # Interior Neumann pixel - not part of solve domain
                # Set φ = 0 as placeholder (won't affect solution in free space)
                row_idx.append(k)
                col_idx.append(k)
                data.append(1.0)
                rhs[k] = 0.0
                continue

            # Free-space pixel: apply Laplacian stencil
            diag = -4.0

            # Up neighbor
            if i - 1 >= 0:
                if dirichlet_mask[i - 1, j]:
                    rhs[k] -= dirichlet_values[i - 1, j]
                elif neumann_mask[i - 1, j]:
                    # Neighbor is Neumann - ghost cell: φ_ghost = φ_here + g*h
                    diag += 1
                    rhs[k] -= neumann_values[i - 1, j] * hy
                else:
                    row_idx.append(k)
                    col_idx.append(idx(i - 1, j))
                    data.append(1.0)
            else:
                # Top edge
                bc = edge_bc["top"]
                if bc["type"] == DIRICHLET:
                    rhs[k] -= top_vals[j]
                else:
                    if h >= 2:
                        row_idx.append(k)
                        col_idx.append(idx(1, j))
                        data.append(1.0)
                    rhs[k] -= (2.0 * hy * top_slope[j])

            # Down neighbor
            if i + 1 < h:
                if dirichlet_mask[i + 1, j]:
                    rhs[k] -= dirichlet_values[i + 1, j]
                elif neumann_mask[i + 1, j]:
                    diag += 1
                    rhs[k] -= neumann_values[i + 1, j] * hy
                else:
                    row_idx.append(k)
                    col_idx.append(idx(i + 1, j))
                    data.append(1.0)
            else:
                # Bottom edge
                bc = edge_bc["bottom"]
                if bc["type"] == DIRICHLET:
                    rhs[k] -= bottom_vals[j]
                else:
                    if h >= 2:
                        row_idx.append(k)
                        col_idx.append(idx(h - 2, j))
                        data.append(1.0)
                    rhs[k] -= (2.0 * hy * bottom_slope[j])

            # Left neighbor
            if j - 1 >= 0:
                if dirichlet_mask[i, j - 1]:
                    rhs[k] -= dirichlet_values[i, j - 1]
                elif neumann_mask[i, j - 1]:
                    diag += 1
                    rhs[k] -= neumann_values[i, j - 1] * hx
                else:
                    row_idx.append(k)
                    col_idx.append(idx(i, j - 1))
                    data.append(1.0)
            else:
                # Left edge
                bc = edge_bc["left"]
                if bc["type"] == DIRICHLET:
                    rhs[k] -= left_vals[i]
                else:
                    if w >= 2:
                        row_idx.append(k)
                        col_idx.append(idx(i, 1))
                        data.append(1.0)
                    rhs[k] -= (2.0 * hx * left_slope[i])

            # Right neighbor
            if j + 1 < w:
                if dirichlet_mask[i, j + 1]:
                    rhs[k] -= dirichlet_values[i, j + 1]
                elif neumann_mask[i, j + 1]:
                    diag += 1
                    rhs[k] -= neumann_values[i, j + 1] * hx
                else:
                    row_idx.append(k)
                    col_idx.append(idx(i, j + 1))
                    data.append(1.0)
            else:
                # Right edge
                bc = edge_bc["right"]
                if bc["type"] == DIRICHLET:
                    rhs[k] -= right_vals[i]
                else:
                    if w >= 2:
                        row_idx.append(k)
                        col_idx.append(idx(i, w - 2))
                        data.append(1.0)
                    rhs[k] -= (2.0 * hx * right_slope[i])

            row_idx.append(k)
            col_idx.append(k)
            data.append(diag)

    A = coo_matrix((data, (row_idx, col_idx)), shape=(N, N), dtype=np.float64).tocsr()
    ml = smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=30)
    phi_flat = ml.solve(rhs, tol=1e-10, maxiter=500)
    return phi_flat.reshape((h, w))


def _solve_poisson_with_flux(
    dirichlet_mask: np.ndarray,
    dirichlet_values: np.ndarray,
    edge_bc: Dict[str, Dict[str, Any]],
    hx: float,
    hy: float,
    rhs_source: np.ndarray,
) -> np.ndarray:
    """Solve Δφ = rhs_source with Dirichlet mask/values and optional Neumann slopes."""
    h, w = dirichlet_mask.shape
    N = h * w
    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    rhs = -rhs_source.astype(np.float64).ravel()

    def idx(i: int, j: int) -> int:
        return j + i * w

    top_vals = _edge_array(edge_bc["top"]["value"], w)
    bottom_vals = _edge_array(edge_bc["bottom"]["value"], w)
    left_vals = _edge_array(edge_bc["left"]["value"], h)
    right_vals = _edge_array(edge_bc["right"]["value"], h)

    top_slope = _edge_array(edge_bc["top"].get("slope", 0.0), w)
    bottom_slope = _edge_array(edge_bc["bottom"].get("slope", 0.0), w)
    left_slope = _edge_array(edge_bc["left"].get("slope", 0.0), h)
    right_slope = _edge_array(edge_bc["right"].get("slope", 0.0), h)

    for i in range(h):
        for j in range(w):
            k = idx(i, j)
            if dirichlet_mask[i, j]:
                row_idx.append(k)
                col_idx.append(k)
                data.append(1.0)
                rhs[k] = dirichlet_values[i, j]
                continue

            diag = -4.0

            # Up
            if i - 1 >= 0:
                if dirichlet_mask[i - 1, j]:
                    rhs[k] -= dirichlet_values[i - 1, j]
                else:
                    row_idx.append(k)
                    col_idx.append(idx(i - 1, j))
                    data.append(1.0)
            else:
                bc = edge_bc["top"]
                if bc["type"] == DIRICHLET:
                    rhs[k] -= top_vals[j]
                else:
                    if h >= 2:
                        row_idx.append(k)
                        col_idx.append(idx(1, j))
                        data.append(1.0)
                    rhs[k] -= (2.0 * hy * top_slope[j])

            # Down
            if i + 1 < h:
                if dirichlet_mask[i + 1, j]:
                    rhs[k] -= dirichlet_values[i + 1, j]
                else:
                    row_idx.append(k)
                    col_idx.append(idx(i + 1, j))
                    data.append(1.0)
            else:
                bc = edge_bc["bottom"]
                if bc["type"] == DIRICHLET:
                    rhs[k] -= bottom_vals[j]
                else:
                    if h >= 2:
                        row_idx.append(k)
                        col_idx.append(idx(h - 2, j))
                        data.append(1.0)
                    rhs[k] -= (2.0 * hy * bottom_slope[j])

            # Left
            if j - 1 >= 0:
                if dirichlet_mask[i, j - 1]:
                    rhs[k] -= dirichlet_values[i, j - 1]
                else:
                    row_idx.append(k)
                    col_idx.append(idx(i, j - 1))
                    data.append(1.0)
            else:
                bc = edge_bc["left"]
                if bc["type"] == DIRICHLET:
                    rhs[k] -= left_vals[i]
                else:
                    if w >= 2:
                        row_idx.append(k)
                        col_idx.append(idx(i, 1))
                        data.append(1.0)
                    rhs[k] -= (2.0 * hx * left_slope[i])

            # Right
            if j + 1 < w:
                if dirichlet_mask[i, j + 1]:
                    rhs[k] -= dirichlet_values[i, j + 1]
                else:
                    row_idx.append(k)
                    col_idx.append(idx(i, j + 1))
                    data.append(1.0)
            else:
                bc = edge_bc["right"]
                if bc["type"] == DIRICHLET:
                    rhs[k] -= right_vals[i]
                else:
                    if w >= 2:
                        row_idx.append(k)
                        col_idx.append(idx(i, w - 2))
                        data.append(1.0)
                    rhs[k] -= (2.0 * hx * right_slope[i])

            row_idx.append(k)
            col_idx.append(k)
            data.append(diag)

    A = coo_matrix((data, (row_idx, col_idx)), shape=(N, N), dtype=np.float64).tocsr()
    ml = smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=30)
    phi_flat = ml.solve(rhs, tol=1e-10, maxiter=500)
    return phi_flat.reshape((h, w))


def _build_object_masks(
    project: Any,
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build separate masks for each BC type from boundary objects.

    Returns:
        (phi_mask, phi_values, w_mask, w_values, neumann_mask, neumann_values)
        - phi_mask/values: Dirichlet mask for φ-solve (all objects)
        - w_mask/values: Dirichlet mask for w-solve (with Laplacian values for Simply Supported)
        - neumann_mask/values: Neumann mask for φ-solve (Flux objects only)
    """
    boundary_objects = project.boundary_objects

    phi_mask = np.zeros((grid_h, grid_w), dtype=bool)
    phi_values = np.zeros((grid_h, grid_w), dtype=float)
    w_mask = np.zeros((grid_h, grid_w), dtype=bool)
    w_values = np.zeros((grid_h, grid_w), dtype=float)
    neumann_mask = np.zeros((grid_h, grid_w), dtype=bool)
    neumann_values = np.zeros((grid_h, grid_w), dtype=float)

    if not boundary_objects:
        return phi_mask, phi_values, w_mask, w_values, neumann_mask, neumann_values

    # Get domain dimensions and margin from project
    if hasattr(project, 'domain_size'):
        domain_w, domain_h = project.domain_size
        grid_scale_x = grid_w / domain_w if domain_w > 0 else 1.0
        grid_scale_y = grid_h / domain_h if domain_h > 0 else 1.0
    else:
        grid_scale_x = 1.0
        grid_scale_y = 1.0

    margin_x, margin_y = project.margin if hasattr(project, 'margin') else (0, 0)

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

        mask_slice = obj_mask[my0:my1, mx0:mx1]
        mask_bool = mask_slice > 0.5

        # Get BC type and parameters
        bc_type = obj.params.get('bc_type', BIHARMONIC_CLAMPED)
        voltage = obj.params.get('voltage', 0.0)
        laplacian = obj.params.get('laplacian', 0.0)
        flux = obj.params.get('neumann_flux', 0.0)

        if bc_type == BIHARMONIC_CLAMPED:
            # Clamped: φ = V, ∂ₙφ ≈ 0 (use band extension later)
            # For now, just mark the base mask; band extension happens in main function
            phi_mask[y0:y1, x0:x1] |= mask_bool
            phi_values[y0:y1, x0:x1] = np.where(
                mask_bool, voltage, phi_values[y0:y1, x0:x1]
            )
            w_mask[y0:y1, x0:x1] |= mask_bool
            # w = 0 for clamped (already zeros)

        elif bc_type == BIHARMONIC_SIMPLY_SUPPORTED:
            # Simply Supported: φ = V, ∇²φ = g (specify Laplacian)
            # No band extension needed
            phi_mask[y0:y1, x0:x1] |= mask_bool
            phi_values[y0:y1, x0:x1] = np.where(
                mask_bool, voltage, phi_values[y0:y1, x0:x1]
            )
            w_mask[y0:y1, x0:x1] |= mask_bool
            w_values[y0:y1, x0:x1] = np.where(
                mask_bool, laplacian, w_values[y0:y1, x0:x1]
            )

        elif bc_type == BIHARMONIC_FLUX:
            # Flux: ∂ₙφ = g (specify normal flux)
            # Object is cut out of domain (Neumann), not Dirichlet
            # DO NOT add to phi_mask - the object boundary has Neumann BC, not Dirichlet
            w_mask[y0:y1, x0:x1] |= mask_bool
            # w = 0 for flux BC (we're not specifying Laplacian)
            neumann_mask[y0:y1, x0:x1] |= mask_bool
            neumann_values[y0:y1, x0:x1] = np.where(
                mask_bool, flux, neumann_values[y0:y1, x0:x1]
            )

    return phi_mask, phi_values, w_mask, w_values, neumann_mask, neumann_values


def solve_biharmonic(project: Any) -> dict[str, np.ndarray]:
    """
    Solve Δ²φ = 0 using a mixed formulation with two Poisson solves.

    Boundary handling for interior objects:
    - Clamped: φ = V, ∂ₙφ ≈ 0 (uses band extension trick)
    - Simply Supported: φ = V, ∇²φ = g (specify Laplacian at boundary)
    - Flux: φ = V, ∂ₙφ = g (specify normal derivative)

    Domain edges support φ (Dirichlet), Δφ (Laplacian), and optional ∂nφ (slope).
    """
    boundary_objects = project.boundary_objects
    grid_h, grid_w = project.shape

    if grid_h == 0 or grid_w == 0:
        return {"phi": np.zeros((grid_h, grid_w), dtype=np.float32)}

    # Physical spacing for scaling and slope application
    if hasattr(project, "domain_size"):
        domain_w, domain_h = project.domain_size
        hx = domain_w / grid_w if grid_w > 0 else 1.0
        hy = domain_h / grid_h if grid_h > 0 else 1.0
    else:
        hx = hy = 1.0

    # Build masks for each BC type
    phi_mask, phi_values, w_mask, w_values, neumann_mask, neumann_values = \
        _build_object_masks(project, grid_h, grid_w)

    # Apply band extension only for Clamped objects
    # We need to identify which parts of phi_mask are from Clamped objects
    # For simplicity, apply band extension to the entire mask, then the w_values
    # will override for Simply Supported objects
    has_clamped = False
    for obj in boundary_objects:
        if obj.params.get('bc_type', BIHARMONIC_CLAMPED) == BIHARMONIC_CLAMPED:
            has_clamped = True
            break

    if has_clamped:
        # Band extension for clamped objects (approximate ∂ₙφ ≈ 0)
        phi_mask_extended, phi_values_extended = _extend_dirichlet_band(phi_mask, phi_values, steps=2)
        # Also extend w_mask for clamped objects
        w_mask_extended, _ = _extend_dirichlet_band(w_mask, w_values, steps=2)
    else:
        phi_mask_extended = phi_mask
        phi_values_extended = phi_values
        w_mask_extended = w_mask

    # Save conductor mask for LIC blocking (all objects where field is ~zero)
    # Include both Dirichlet (extended) and Neumann objects
    conductor_dirichlet_mask = phi_mask_extended | neumann_mask

    # Edge BCs from project.bc (resolved upstream)
    bc_map = getattr(project, "bc", {}) or {}
    edge_bc = _edge_bc_from_map(bc_map)
    phi_mask_final, phi_values_final = _apply_edge_dirichlet(phi_mask_extended, phi_values_extended, edge_bc)

    # Mixed formulation: w = Δφ. Solve Δw = 0 with boundary Dirichlet = provided laplacian.
    lap_top = _edge_array(edge_bc["top"]["laplacian"], grid_w)
    lap_bottom = _edge_array(edge_bc["bottom"]["laplacian"], grid_w)
    lap_left = _edge_array(edge_bc["left"]["laplacian"], grid_h)
    lap_right = _edge_array(edge_bc["right"]["laplacian"], grid_h)

    w_dirichlet_mask = np.zeros((grid_h, grid_w), dtype=bool)
    w_dirichlet_vals = np.zeros((grid_h, grid_w), dtype=float)

    # Edge Dirichlet for w
    w_dirichlet_mask[0, :] = True
    w_dirichlet_vals[0, :] = lap_top
    w_dirichlet_mask[grid_h - 1, :] = True
    w_dirichlet_vals[grid_h - 1, :] = lap_bottom
    w_dirichlet_mask[:, 0] = True
    w_dirichlet_vals[:, 0] = lap_left
    w_dirichlet_mask[:, grid_w - 1] = True
    w_dirichlet_vals[:, grid_w - 1] = lap_right

    # Add object w masks (with Laplacian values for Simply Supported)
    w_dirichlet_mask |= w_mask_extended
    # w_values has non-zero values for Simply Supported objects
    w_dirichlet_vals = np.where(w_mask_extended, w_values, w_dirichlet_vals)

    w_field = solve_poisson_system(
        w_dirichlet_mask,
        w_dirichlet_vals,
        boundary_top=DIRICHLET,
        boundary_bottom=DIRICHLET,
        boundary_left=DIRICHLET,
        boundary_right=DIRICHLET,
        charge_density=np.zeros((grid_h, grid_w), dtype=float),
    )

    # Second solve: Δφ = w with Dirichlet φ on objects and edges
    # For Flux objects, use Neumann handling
    has_neumann = neumann_mask.any()

    if has_neumann:
        # Use the Neumann-aware solver
        phi_field = _solve_poisson_with_interior_neumann(
            phi_mask_final,
            phi_values_final,
            neumann_mask,
            neumann_values,
            edge_bc,
            hx,
            hy,
            rhs_source=w_field.astype(float),
        )
    else:
        # Original solver (no interior Neumann)
        phi_field = _solve_poisson_with_flux(
            phi_mask_final,
            phi_values_final,
            edge_bc,
            hx,
            hy,
            rhs_source=w_field.astype(float),
        )

    phi = phi_field.astype(np.float32)

    return {"phi": phi, "dirichlet_mask": conductor_dirichlet_mask}


BIHARMONIC_PDE = PDEDefinition(
    name="biharmonic",
    display_name="Biharmonic Flow (Stokes)",
    description="Solve biharmonic equation (clamped plate / Stokes flow)",
    solve=solve_biharmonic,
    extract_lic_field=extract_electric_field,  # Same E = -grad(phi)
    boundary_params=[],  # Using boundary_fields instead
    boundary_fields=[
        BCField(
            name="bc_type",
            display_name="Boundary Type",
            field_type="enum",
            default=BIHARMONIC_CLAMPED,
            choices=[
                ("Clamped (∂ₙφ=0)", BIHARMONIC_CLAMPED),
                ("Simply Supported (∇²φ=g)", BIHARMONIC_SIMPLY_SUPPORTED),
                ("Flux (∂ₙφ=g)", BIHARMONIC_FLUX),
            ],
            description="Clamped=zero slope, Simply Supported=specify curvature, Flux=specify normal derivative"
        ),
        BCField(
            name="voltage",
            display_name="Potential (φ)",
            field_type="float",
            default=0.0,
            min_value=-1.0,
            max_value=1.0,
            description="Stream function value at the boundary",
        ),
        BCField(
            name="laplacian",
            display_name="Laplacian (∇²φ)",
            field_type="float",
            default=0.0,
            min_value=-1.0,
            max_value=1.0,
            description="Laplacian (curvature) at the boundary",
            visible_when={"bc_type": BIHARMONIC_SIMPLY_SUPPORTED},
        ),
        BCField(
            name="neumann_flux",
            display_name="Normal Flux (∂ₙφ)",
            field_type="float",
            default=0.0,
            min_value=-1.0,
            max_value=1.0,
            description="Normal derivative of potential at boundary",
            visible_when={"bc_type": BIHARMONIC_FLUX},
        ),
    ],
    bc_fields=[
        BCField(
            name="type",
            display_name="Boundary Type",
            field_type="enum",
            default=DIRICHLET,
            choices=[("Dirichlet (fixed)", DIRICHLET), ("Neumann (free)", NEUMANN)],
            description="Dirichlet = fixed potential, Neumann = zero normal derivative."
        ),
        BCField(
            name="value",
            display_name="Value",
            field_type="float",
            default=0.0,
            min_value=-10.0,
            max_value=10.0,
            description="Fixed potential value along the edge.",
            visible_when={"type": DIRICHLET},
        ),
        BCField(
            name="slope",
            display_name="Normal Slope",
            field_type="float",
            default=0.0,
            min_value=-10.0,
            max_value=10.0,
            description="Outward normal derivative at the edge.",
            visible_when={"type": NEUMANN},
        ),
        BCField(
            name="laplacian",
            display_name="Laplacian",
            field_type="float",
            default=0.0,
            min_value=-10.0,
            max_value=10.0,
            description="Laplacian value at edge (for mixed biharmonic solve)."
        ),
    ],
)
