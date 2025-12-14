# Elliptica Code Review Guide

This document identifies core separable components for isolated code review by independent agents.

## Core Components

- **PDE Physics Engine** (`elliptica/pde/`, `elliptica/poisson.py`, `elliptica/field_pde.py`)
- **LIC Visualization** (`elliptica/lic.py`, `elliptica/pipeline.py`, `elliptica/render.py`)
- **Colorization System** (`elliptica/colorspace/`, `elliptica/postprocess/`)
- **GPU Acceleration** (`elliptica/gpu/`)
- **Expression Parser** (`elliptica/expr/`)
- **State Management** (`elliptica/types.py`, `elliptica/app/core.py`, `elliptica/app/actions.py`)
- **Project Serialization** (`elliptica/serialization.py`)
- **UI Controllers** (`elliptica/ui/dpg/`)

---

## Agent Review Instructions

### 1. PDE Physics Engine

Review the PDE solver system in `elliptica/pde/` (registry, base class, implementations for Poisson/Biharmonic/Eikonal), the low-level Poisson solver in `elliptica/poisson.py`, and the orchestrator in `elliptica/field_pde.py`. Focus on numerical correctness of the solvers, proper handling of boundary conditions (Dirichlet/Neumann), the plugin architecture via `PDERegistry`, memory efficiency with large grids, and whether the multi-scale solving approach (solve at lower resolution then upsample) introduces artifacts. Check for potential numerical instabilities, division-by-zero edge cases, and whether sparse matrix construction is optimal.

### 2. LIC Visualization

Review the Line Integral Convolution system in `elliptica/lic.py`, the render orchestration in `elliptica/pipeline.py`, and supporting utilities in `elliptica/render.py`. Focus on correct integration with the rLIC C++ library, proper handling of masked regions (conductors should block streamlines), edge cases with zero-magnitude vectors, memory management for large renders, and whether the noise texture seeding is deterministic. Verify that render metadata (margins, offsets, crop coordinates) is computed correctly and that the pipeline properly handles resolution scaling.

### 3. Colorization System

Review the color processing in `elliptica/colorspace/` (OKLCH conversions, gamut mapping, expression-based color mapping) and `elliptica/postprocess/` (region overlays, mask handling). Focus on perceptual correctness of OKLCH conversions, gamut mapping strategies (clipping vs compression), expression evaluation safety, and whether out-of-gamut colors are handled gracefully. Check that per-region colorization with alpha blending produces correct results and that the preset system is extensible.

### 4. GPU Acceleration

Review the GPU acceleration layer in `elliptica/gpu/` including device detection, tensor operations, and accelerated post-processing. Focus on correct fallback behavior (CUDA → MPS → CPU), memory management and tensor lifecycle, whether operations produce identical results on CPU vs GPU, and proper handling of device transfers. Check for potential memory leaks, unnecessary CPU↔GPU transfers, and whether the lazy initialization pattern is robust.

### 5. Expression Parser

Review the expression system in `elliptica/expr/` (parser, compiler, built-in functions, error handling). Focus on parsing correctness for mathematical expressions, security of the evaluation (no arbitrary code execution), performance of compiled expressions on large arrays, and error message quality. Verify the safety limits (max_depth, max_nodes) prevent DoS, the percentile caching works correctly, and all advertised functions (normalize, magnitude, blur, etc.) behave as documented.

### 6. State Management

Review the data model in `elliptica/types.py`, the central state container in `elliptica/app/core.py`, and mutation helpers in `elliptica/app/actions.py`. Focus on thread-safety (state is accessed from UI and render threads), correctness of dirty flag propagation, whether state mutations are atomic where needed, and whether the conductor/boundary object abstraction cleanly supports multiple PDE types. Check for potential race conditions and verify the render cache invalidation logic is correct.

### 7. Project Serialization

Review the ZIP-based project format in `elliptica/serialization.py`. Focus on backwards compatibility (can old projects load in new versions), forward compatibility (graceful handling of unknown fields), data integrity (are masks saved/loaded losslessly), and security (path traversal attacks in ZIP extraction). Verify all state is round-tripped correctly including per-conductor settings, color configurations, and PDE-specific parameters. Check error handling for corrupted files.

### 8. UI Controllers

Review the DearPyGUI-based UI in `elliptica/ui/dpg/`. Key files: `app.py` (main orchestrator), `canvas_controller.py` (mouse/keyboard input), `canvas_renderer.py` (display), `postprocessing_panel.py` (largest panel), `boundary_controls_panel.py`, `file_io_controller.py`, `image_export_controller.py`, and `render_modal.py`. Focus on event handling correctness, proper blocking of input when dialogs/menus are open, thread-safe state access, texture lifecycle management, and whether the controller separation is clean. Check for UI state inconsistencies and verify all user-facing operations have appropriate feedback.
