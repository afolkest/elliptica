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
    boundary_type=DIRICHLET
    ):
    """ 

    Explanation: 

    In units of dx=1 we have
    (laplacian f)[i,j] = -4 f[i,j] + f[i+1,j] + f[i-1,j] + f[i,j+1] + f[i, j-1]
    If (i, j) is a conductor point, we simply enforce phi[i, j] = V[i, j] at this point
    If (i, j) is an interior point next to a dirichelet pt, we move the fix conductor values to the rhs 
    """

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

            if dirichlet_mask[i, j]:
                #enforce phi = V
                row_index[n_nonzero] = k
                col_index[n_nonzero] = k
                almost_laplacian[n_nonzero] = 1.0 
                n_nonzero += 1 

                rhs[k] = dirichlet_voltage[i, j]
            else:
                diagonal = -4.0 
                rhs[k] = -charge_density[i, j]

                #off-diagonal entries
                for di, dj in [(-1,0), (1, 0), (0, -1), (0, 1)]:
                    ii, jj = i + di, j + dj 
                    kk = jj + ii * width 

                    if 0 <= ii < height and 0 <= jj < width:
                        if dirichlet_mask[ii, jj]:
                            rhs[k] -= dirichlet_voltage[ii, jj]
                        else: 
                            row_index[n_nonzero] = k 
                            col_index[n_nonzero] = kk
                            almost_laplacian[n_nonzero] = 1.0 
                            n_nonzero += 1
                    elif boundary_type == NEUMANN:
                        diagonal += 1

                row_index[n_nonzero] = k
                col_index[n_nonzero] = k
                almost_laplacian[n_nonzero] = diagonal 
                n_nonzero += 1 


    return row_index[:n_nonzero], col_index[:n_nonzero], almost_laplacian[:n_nonzero], rhs

def solve_poisson_system(
    dirichlet_mask,
    dirichlet_values,
    tol = SOLVER_TOL,
    maxiter = 2000,
    boundary_type=DIRICHLET,
    charge_density=None
):

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
        boundary_type 
    )

    N = height * width
    A = coo_matrix((almost_laplacian, (row_index, col_index)), shape=(N, N), dtype=PRECISION).tocsr()
    multilevel_solver = smoothed_aggregation_solver(
        A,
        strength="symmetric",
        max_coarse=10,
        max_levels=10
    )
    preconditioner  = multilevel_solver.aspreconditioner() #preconditioner

    phi_flat, info = cg(A, rhs.astype(PRECISION), M=preconditioner, rtol=tol, maxiter=maxiter)

    if info > 0:
        print(f"Warning: poisson solve not converged")
    elif info < 0:
        raise RuntimeError(f"CG failed with error code {info}")
    
    #phi_flat = multilevel_solver.solve(rhs, tol=1e-5, maxiter=100, cycle='V') #slightly faster

    return phi_flat.reshape(height, width)