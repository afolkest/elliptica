
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from flowcol.types import Project, BoundaryObject
from flowcol.pde.poisson_pde import solve_poisson, extract_electric_field
from flowcol.pde.biharmonic_pde import solve_biharmonic
from flowcol.pde.register import register_all_pdes

def create_test_project():
    project = Project(canvas_resolution=(256, 256))
    
    # Create a circular conductor in the center
    size = 256
    y, x = np.ogrid[:size, :size]
    cy, cx = size / 2.0, size / 2.0
    radius = size / 4.0
    mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
    
    obj = BoundaryObject(mask=mask.astype(np.float32), voltage=1.0)
    project.boundary_objects.append(obj)
    
    return project, mask

def get_boundary_normals(mask):
    """Compute normals at the boundary of the mask."""
    from scipy.ndimage import sobel
    sx = sobel(mask)
    sy = sobel(mask)
    mag = np.sqrt(sx**2 + sy**2)
    mag[mag == 0] = 1.0
    return sx / mag, sy / mag, mag > 0

def verify():
    print("Registering PDEs...")
    register_all_pdes()
    
    print("Creating test project...")
    project, mask = create_test_project()
    
    print("Solving Poisson (Reference)...")
    sol_p = solve_poisson(project)
    ex_p, ey_p = extract_electric_field(sol_p, project)
    
    print("Solving Biharmonic (New)...")
    sol_b = solve_biharmonic(project)
    ex_b, ey_b = extract_electric_field(sol_b, project)
    
    # Analyze boundary behavior
    nx, ny, boundary_mask = get_boundary_normals(mask.astype(float))
    
    # Sample field at boundary
    # We look at pixels just outside the mask?
    # The gradient is defined everywhere.
    # Let's look at the dot product E . n at the boundary pixels.
    
    # Filter for boundary
    b_indices = np.where(boundary_mask)
    
    # Poisson: E should be parallel to N (E . N is large, E x N is small)
    # Biharmonic: E should be perpendicular to N (E . N is small, E x N is large)
    
    # Normalize fields
    mag_p = np.sqrt(ex_p**2 + ey_p**2)
    mag_p[mag_p == 0] = 1.0
    ex_p_n, ey_p_n = ex_p / mag_p, ey_p / mag_p
    
    mag_b = np.sqrt(ex_b**2 + ey_b**2)
    mag_b[mag_b == 0] = 1.0
    ex_b_n, ey_b_n = ex_b / mag_b, ey_b / mag_b
    
    # Dot products
    dot_p = np.abs(ex_p_n * nx + ey_p_n * ny)
    dot_b = np.abs(ex_b_n * nx + ey_b_n * ny)
    
    avg_dot_p = np.mean(dot_p[boundary_mask])
    avg_dot_b = np.mean(dot_b[boundary_mask])
    
    print(f"\nResults (Alignment with Normal):")
    print(f"Poisson (Expect High): {avg_dot_p:.4f}")
    print(f"Biharmonic (Expect Low): {avg_dot_b:.4f}")
    
    if avg_dot_b < avg_dot_p:
        print("\nSUCCESS: Biharmonic field is more tangential (Stokes-like) than Poisson.")
    else:
        print("\nFAILURE: Biharmonic field is NOT more tangential.")

if __name__ == "__main__":
    verify()
