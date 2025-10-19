import numba
import numpy as np

@numba.jit(nopython=True, cache=True)
def _build_poisson_system(
    height, 
    width, 
    dirichlet_mask,
    dirichlet_voltage,
    charge_density, 
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
    almost_laplacian = np.empty(total_nonzero, dtype=np.float64)

    #rhs, equalling negative charge density except at or adjacent to conductor 
    rhs = np.empty(N, dtype=np.float64)

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
                row_index[n_nonzero] = k
                col_index[n_nonzero] = k
                almost_laplacian[n_nonzero] = -4.0 #diagonal entry
                n_nonzero += 1 

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

    return row_index[:n_nonzero], col_index[:n_nonzero], almost_laplacian[:n_nonzero], rhs
