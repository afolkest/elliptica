from dataclasses import dataclass
from typing import Any, Optional
import numba
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from pyamg import smoothed_aggregation_solver

DIRICHLET = 0
NEUMANN = 1

PRECISION = np.float64 
SOLVER_TOL = 1e-5  

@numba.jit(nopython=True, cache=True)
def _build_poisson_system(
    height,
    width,
    dirichlet_mask,
    dirichlet_voltage,
    charge_density,
    boundary_top=DIRICHLET,
    boundary_bottom=DIRICHLET,
    boundary_left=DIRICHLET,
    boundary_right=DIRICHLET,
    neumann_mask=None,
    ):
    """
    Build the Poisson system matrix and RHS.

    In units of dx=1 we have
    (laplacian f)[i,j] = -4 f[i,j] + f[i+1,j] + f[i-1,j] + f[i,j+1] + f[i, j-1]

    Boundary handling:
    - dirichlet_mask: pixels where φ = V (fixed potential)
    - neumann_mask: pixels that are interior obstacles with zero normal flux (∂φ/∂n = 0)
      These are cut out of the domain; adjacent free-space pixels use reflection.
    """
    # Handle None neumann_mask (for backwards compatibility)
    has_neumann = neumann_mask is not None

    N = height * width
    total_nonzero = 5*N #5 point stencil

    row_index = np.empty(total_nonzero, dtype=np.int32) #parametrizes which equation
    col_index = np.empty(total_nonzero, dtype=np.int32)

    #sparse matrix operator, equal to laplacian except at or adjacent to conductors
    almost_laplacian = np.empty(total_nonzero, dtype=PRECISION)

    #rhs, equalling negative charge density except at or adjacent to conductor
    rhs = np.empty(N, dtype=PRECISION)

    n_nonzero = 0

    for i in range(height):
        for j in range(width):
            k = j + i * width

            # Check if this pixel is in a Neumann region (cut out of domain)
            in_neumann = has_neumann and neumann_mask[i, j]

            if dirichlet_mask[i, j]:
                #enforce phi = V
                row_index[n_nonzero] = k
                col_index[n_nonzero] = k
                almost_laplacian[n_nonzero] = 1.0
                n_nonzero += 1
                rhs[k] = dirichlet_voltage[i, j]
            elif in_neumann:
                # Neumann interior pixel - not part of the solve domain
                # Set phi = 0 as a placeholder (won't affect solution in free space)
                row_index[n_nonzero] = k
                col_index[n_nonzero] = k
                almost_laplacian[n_nonzero] = 1.0
                n_nonzero += 1
                rhs[k] = 0.0
            else:
                diagonal = -4.0
                rhs[k] = -charge_density[i, j]

                #off-diagonal entries
                for di, dj in [(-1,0), (1, 0), (0, -1), (0, 1)]:
                    ii, jj = i + di, j + dj
                    kk = jj + ii * width

                    if 0 <= ii < height and 0 <= jj < width:
                        if dirichlet_mask[ii, jj]:
                            # Neighbor is Dirichlet - move to RHS
                            rhs[k] -= dirichlet_voltage[ii, jj]
                        elif has_neumann and neumann_mask[ii, jj]:
                            # Neighbor is in Neumann region - apply reflection (∂φ/∂n = 0)
                            # Ghost value equals this pixel's value, so add 1 to diagonal
                            diagonal += 1
                        else:
                            # Normal interior neighbor
                            row_index[n_nonzero] = k
                            col_index[n_nonzero] = kk
                            almost_laplacian[n_nonzero] = 1.0
                            n_nonzero += 1
                    else:
                        # Out of bounds - determine which boundary and apply its condition
                        if di == -1 and boundary_top == NEUMANN:  # top edge
                            diagonal += 1
                        elif di == 1 and boundary_bottom == NEUMANN:  # bottom edge
                            diagonal += 1
                        elif dj == -1 and boundary_left == NEUMANN:  # left edge
                            diagonal += 1
                        elif dj == 1 and boundary_right == NEUMANN:  # right edge
                            diagonal += 1
                        # DIRICHLET is implicit (φ=0 at boundary) - no change needed

                row_index[n_nonzero] = k
                col_index[n_nonzero] = k
                almost_laplacian[n_nonzero] = diagonal
                n_nonzero += 1


    return row_index[:n_nonzero], col_index[:n_nonzero], almost_laplacian[:n_nonzero], rhs

@dataclass
class PoissonSolverContext:
    """Context holding the pre-built matrix and preconditioner for fast solving."""
    A: Any
    preconditioner: Any
    row_index: np.ndarray
    col_index: np.ndarray
    almost_laplacian: np.ndarray
    shape: tuple[int, int]
    dirichlet_mask: np.ndarray

def build_poisson_solver(
    dirichlet_mask,
    dirichlet_values,
    boundary_top=DIRICHLET,
    boundary_bottom=DIRICHLET,
    boundary_left=DIRICHLET,
    boundary_right=DIRICHLET,
    charge_density=None,
    neumann_mask=None,
) -> PoissonSolverContext:
    """
    Build the system matrix and preconditioner for the Poisson equation.
    Returns a context object that can be used with solve_poisson_fast.

    Args:
        neumann_mask: Optional boolean mask for interior Neumann boundaries.
            Pixels in this mask are treated as insulating obstacles (∂φ/∂n = 0).
    """
    height, width = dirichlet_mask.shape
    if charge_density is None:
        charge_density = np.zeros((height, width), dtype=PRECISION)
    else:
        charge_density = charge_density.astype(PRECISION)

    dirichlet_values = dirichlet_values.astype(PRECISION)

    row_index, col_index, almost_laplacian, rhs = _build_poisson_system(
        height,
        width,
        dirichlet_mask,
        dirichlet_values,
        charge_density,
        boundary_top,
        boundary_bottom,
        boundary_left,
        boundary_right,
        neumann_mask,
    )

    N = height * width
    A = coo_matrix((almost_laplacian, (row_index, col_index)), shape=(N, N), dtype=PRECISION).tocsr()
    multilevel_solver = smoothed_aggregation_solver(
        A,
        strength="symmetric",
        max_coarse=10,
        max_levels=10
    )
    preconditioner = multilevel_solver.aspreconditioner()

    return PoissonSolverContext(
        A=A,
        preconditioner=preconditioner,
        row_index=row_index,
        col_index=col_index,
        almost_laplacian=almost_laplacian,
        shape=(height, width),
        dirichlet_mask=dirichlet_mask
    )

def solve_poisson_fast(
    context: PoissonSolverContext,
    dirichlet_values: np.ndarray,
    charge_density: Optional[np.ndarray] = None,
    tol=SOLVER_TOL,
    maxiter=2000
) -> np.ndarray:
    """
    Solve the Poisson system using a pre-built context.
    Only the RHS (dirichlet_values and charge_density) can change.
    """
    height, width = context.shape
    N = height * width
    
    if charge_density is None:
        charge_density = np.zeros((height, width), dtype=PRECISION)
    
    # Reconstruct RHS efficiently without rebuilding the matrix
    # We need to replicate the RHS logic from _build_poisson_system but in Python/Numpy for speed
    # or just call a simplified Numba function.
    # For now, let's reuse _build_poisson_system but ignore the matrix outputs to ensure correctness,
    # as optimizing RHS construction is a secondary optimization.
    
    # Note: To truly optimize, we should split _build_poisson_system into build_matrix and build_rhs.
    # But for now, let's just use the existing function and discard the matrix parts.
    # The matrix build is the slow part usually, but let's verify.
    # Actually, _build_poisson_system does both. 
    # Let's write a fast RHS builder.
    
    rhs = _build_rhs_only(
        height, width, 
        context.dirichlet_mask, 
        dirichlet_values.astype(PRECISION), 
        charge_density.astype(PRECISION),
        context.almost_laplacian # Pass this if needed, but actually we just need the logic
    )

    phi_flat, info = cg(context.A, rhs, M=context.preconditioner, rtol=tol, maxiter=maxiter)

    if info > 0:
        print(f"Warning: poisson solve not converged")
    elif info < 0:
        raise RuntimeError(f"CG failed with error code {info}")

    return phi_flat.reshape(height, width)

@numba.jit(nopython=True, cache=True)
def _build_rhs_only(height, width, dirichlet_mask, dirichlet_voltage, charge_density, dummy_arg=None):
    """Fast RHS reconstruction."""
    N = height * width
    rhs = np.empty(N, dtype=PRECISION)
    
    for i in range(height):
        for j in range(width):
            k = j + i * width
            
            if dirichlet_mask[i, j]:
                rhs[k] = dirichlet_voltage[i, j]
            else:
                rhs[k] = -charge_density[i, j]
                # Check neighbors for Dirichlet boundaries
                for di, dj in [(-1,0), (1, 0), (0, -1), (0, 1)]:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < height and 0 <= jj < width:
                        if dirichlet_mask[ii, jj]:
                            rhs[k] -= dirichlet_voltage[ii, jj]
    return rhs

def solve_poisson_system(
    dirichlet_mask,
    dirichlet_values,
    tol = SOLVER_TOL,
    maxiter = 2000,
    boundary_top=DIRICHLET,
    boundary_bottom=DIRICHLET,
    boundary_left=DIRICHLET,
    boundary_right=DIRICHLET,
    charge_density=None,
    neumann_mask=None,
):
    """
    Solve the Poisson equation with Dirichlet and optional Neumann boundaries.

    Args:
        neumann_mask: Optional boolean mask for interior Neumann boundaries.
            Pixels in this mask are treated as insulating obstacles (∂φ/∂n = 0).
    """
    context = build_poisson_solver(
        dirichlet_mask,
        dirichlet_values,
        boundary_top,
        boundary_bottom,
        boundary_left,
        boundary_right,
        charge_density,
        neumann_mask,
    )
    return solve_poisson_fast(context, dirichlet_values, charge_density, tol, maxiter)