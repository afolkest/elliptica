# Elliptica

**Physics-based generative art framework**

Elliptica is a modular framework for creating visual art using physics simulations. While it originated as an electrostatics visualizer, it has evolved into a generic engine for **Partial Differential Equation (PDE)** based art.

You draw with physics: place boundary objects (conductors, obstacles, sources) on a canvas, solve a physical system (like the Poisson equation for electric fields), and visualize the resulting vector fields using high-quality Line Integral Convolution (LIC).

![Example renders would go here]

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concept](#core-concept)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Project File Format](#project-file-format)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the GUI (defaults to Electrostatics mode)
python -m elliptica.ui.dpg
```

## Core Concept

Elliptica transforms mathematical physics into visual art through a four-step pipeline:

1.  **Define Geometry** - Import shapes as "Boundary Objects" (formerly Conductors) using PNG masks.
2.  **Select Physics** - Choose a PDE solver from the registry (e.g., Electrostatics).
3.  **Solve System** - The framework solves the chosen PDE (e.g., $\nabla^2\phi = 0$) to compute a vector field.
4.  **Visualize Flow** - Line Integral Convolution (LIC) creates flowing textures that follow the field lines.
5.  **Colorize & Refine** - Apply GPU-accelerated post-processing for artistic effect.

## Features

### Pluggable Physics Engine
-   **Generic PDE System** - Architecture supports arbitrary physics solvers via `PDERegistry`.
-   **Electrostatics (Default)** - Robust Poisson solver using PyAMG-preconditioned conjugate gradient.
-   **Extensible** - Easily add new physics modules (Fluid dynamics, Reaction-Diffusion, etc.) by implementing the `PDEDefinition` interface.

### Rendering & Visualization
-   **High-quality LIC** - Multi-pass line integral convolution using the rLIC library.
-   **GPU Acceleration** - PyTorch MPS/CUDA backend for real-time post-processing.
-   **Anisotropic Blur** - Field-aligned blur effects.

### Interactive UI
-   **Visual Canvas** - Direct manipulation of boundary objects.
-   **Real-time Feedback** - Adjust physics parameters and see results instantly (using low-res preview).
-   **Project Management** - Save/load projects with full state preservation.

## System Requirements

### Python
-   **Python 3.11 or later** (3.12, 3.13 also work)

### Operating System
-   **macOS** (Intel or Apple Silicon)
-   **Linux** (requires X11 or Wayland display)
-   **Windows** 10/11

### GPU (Optional)
GPU acceleration is optional but recommended for real-time post-processing:
-   **macOS**: Apple Silicon with MPS (M1/M2/M3) - automatic
-   **Linux/Windows**: NVIDIA GPU with CUDA - automatic
-   **CPU fallback**: Works without GPU, just slower

### Display
Elliptica is a GUI application and requires a graphical display. Headless environments (SSH without X11, Docker without display) are not supported.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/elliptica.git
cd elliptica

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python -m elliptica.ui.dpg
```

## Usage

### GUI Workflow
1.  **Import Shapes** - Load PNG images to define your boundary objects.
2.  **Configure Physics** - Set values (e.g., Voltage) for each object.
3.  **Render** - Click render to solve the PDE and generate the LIC visualization.
4.  **Post-Process** - Adjust colors, contrast, and blur in real-time.

### Programmatic Usage

Elliptica can be used as a library. The new API supports the generic PDE system:

```python
from elliptica.types import Project, BoundaryObject
from elliptica.field_pde import compute_field_pde
from elliptica.pipeline import perform_render
from elliptica.mask_utils import load_mask_from_png
from elliptica.pde import PDERegistry

# Ensure the desired PDE is active (defaults to 'poisson')
PDERegistry.set_active("poisson")

# Create project
project = Project(canvas_resolution=(1024, 1024))

# Load a boundary object (e.g., a conductor)
mask = load_mask_from_png("shape.png")
obj = BoundaryObject(mask=mask, voltage=1.0, position=(512, 512))
project.boundary_objects.append(obj)

# Compute field using the active PDE solver
# Returns solution dict (e.g. {'phi': ...}) and vector field (ex, ey)
solution, (ex, ey) = compute_field_pde(
    project,
    multiplier=2.0,
    margin=(0.1, 0.1)
)

# Render LIC visualization
result = perform_render(
    project,
    multiplier=2.0,
    num_passes=2,
    streamlength_factor=0.05
)
```

## Architecture

Elliptica has evolved into a layered architecture separating the generic physics engine from the rendering and UI layers.

### Core Components

```
elliptica/
   pde/                 # Pluggable Physics System
      base.py           # PDEDefinition interface
      registry.py       # Global PDE registry
      poisson_pde.py    # Electrostatics implementation
   
   field_pde.py         # Generic field computation orchestrator
   pipeline.py          # Render pipeline (PDE solve -> LIC -> Post-process)
   types.py             # Core data structures (Project, BoundaryObject)
   
   # Legacy/Support
   poisson.py           # Low-level Poisson solver
   field.py             # Legacy wrapper (deprecated)
```

### Key Concepts

1.  **BoundaryObject**: A generic entity with a shape (mask) and a value. In Electrostatics, this represents a Conductor with Voltage. In other PDEs, it could represent an Obstacle or Heat Source.
2.  **PDEDefinition**: An interface defining how to `solve()` a system and `extract_lic_field()` from the solution.
3.  **Render Pipeline**: A functional pipeline that takes a `Project` state, runs the active PDE solver, and feeds the result into the LIC renderer.

## Project File Format

Projects are saved as `.elliptica` ZIP archives. The format has been updated to support generic metadata:

```json
{
  "project": {
    "pde_type": "poisson",
    "pde_params": { ... }
  },
  "conductors": [
    {
      "voltage": 1.0,  # Primary value
      "position": [512.0, 512.0],
      "scale_factor": 1.0
    }
  ]
}
```

## Troubleshooting

### "No display found" / HeadlessEnvironmentError

Elliptica requires a graphical display. This error occurs when running on:
-   Remote servers via SSH (without X11 forwarding)
-   Docker containers without display access
-   CI environments

**Solutions:**
-   SSH with X11 forwarding: `ssh -X user@host`
-   Docker: Mount X11 socket or use `xvfb-run`
-   Use a machine with a physical display

### "Failed to create display window"

Graphics driver or display server issue.

**Solutions:**
-   Update graphics drivers
-   Check display server is running (X11/Wayland on Linux)
-   Try a different terminal or restart the display manager

### GPU Not Detected / Slow Performance

Elliptica auto-detects GPU (MPS on Apple Silicon, CUDA on NVIDIA). If detection fails, it falls back to CPU.

**Check GPU status:**
```python
from elliptica.gpu import GPUContext
print(GPUContext.device())  # Should show 'mps', 'cuda', or 'cpu'
```

**If showing 'cpu' unexpectedly:**
-   macOS: Ensure you have Apple Silicon (M1/M2/M3), not Intel
-   Linux/Windows: Install CUDA toolkit and PyTorch with CUDA support
-   Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### Project File Won't Load

**"Schema version X not supported":** Project was saved with a different Elliptica version. Re-export from the original version or recreate the project.

**Corrupt file:** If the `.elliptica` file is damaged, it cannot be recovered. Keep backups of important projects.

### Import Errors / Module Not Found

Ensure you're in the correct virtual environment:
```bash
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Development

### Philosophy
-   **Functional Backend**: Physics and rendering are pure functions.
-   **Modular Frontend**: UI controllers are single-responsibility.

### Contributing
To add a new physics engine:
1.  Implement `PDEDefinition` in `elliptica/pde/`.
2.  Register it in `elliptica/pde/__init__.py`.
3.  Ensure it returns a vector field `(ex, ey)` for LIC visualization.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE) for details.

## Credits

-   **rLIC**: Line Integral Convolution library
-   **PyAMG**: Algebraic multigrid solver
-   **Dear PyGui**: Modern GUI framework
