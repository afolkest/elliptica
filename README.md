# FlowCol

**Physics-based generative art framework**

FlowCol is a modular framework for creating visual art using physics simulations. While it originated as an electrostatics visualizer, it has evolved into a generic engine for **Partial Differential Equation (PDE)** based art.

You draw with physics: place boundary objects (conductors, obstacles, sources) on a canvas, solve a physical system (like the Poisson equation for electric fields), and visualize the resulting vector fields using high-quality Line Integral Convolution (LIC).

![Example renders would go here]

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concept](#core-concept)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Project File Format](#project-file-format)
- [Performance](#performance)
- [Development](#development)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the GUI (defaults to Electrostatics mode)
python -m flowcol.ui.dpg
```

## Core Concept

FlowCol transforms mathematical physics into visual art through a four-step pipeline:

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

## Installation

### Requirements
-   Python 3.10+
-   macOS with Apple Silicon (recommended for MPS acceleration) or CUDA-compatible GPU

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### GUI Workflow
1.  **Import Shapes** - Load PNG images to define your boundary objects.
2.  **Configure Physics** - Set values (e.g., Voltage) for each object.
3.  **Render** - Click render to solve the PDE and generate the LIC visualization.
4.  **Post-Process** - Adjust colors, contrast, and blur in real-time.

### Programmatic Usage

FlowCol can be used as a library. The new API supports the generic PDE system:

```python
from flowcol.types import Project, BoundaryObject
from flowcol.field_pde import compute_field_pde
from flowcol.pipeline import perform_render
from flowcol.mask_utils import load_mask_from_png
from flowcol.pde import PDERegistry

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

FlowCol has evolved into a layered architecture separating the generic physics engine from the rendering and UI layers.

### Core Components

```
flowcol/
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

Projects are saved as `.flowcol` ZIP archives. The format has been updated to support generic metadata:

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

## Development

### Philosophy
-   **Functional Backend**: Physics and rendering are pure functions.
-   **Modular Frontend**: UI controllers are single-responsibility.
-   **No Defensive Programming**: Assume correct usage for maximum performance.

### Contributing
To add a new physics engine:
1.  Implement `PDEDefinition` in `flowcol/pde/`.
2.  Register it in `flowcol/pde/__init__.py`.
3.  Ensure it returns a vector field `(ex, ey)` for LIC visualization.

## License

[License Information]

## Credits

-   **rLIC**: Line Integral Convolution library
-   **PyAMG**: Algebraic multigrid solver
-   **Dear PyGui**: Modern GUI framework
