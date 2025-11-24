#!/usr/bin/env python3

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "LineIntegralConvolutions" / "src"))

from flowcol.types import Project, Conductor
from flowcol.field_pde import compute_field_pde
from flowcol.render import (
    compute_lic,
    apply_gaussian_highpass,
    downsample_lic,
)
from flowcol.mask_utils import load_alpha
from vegtamr.lic import compute_lic_with_postprocessing


ASSET = ROOT / "assets" / "masks" / "disciples_shell.png"


def prepare_project(mask: np.ndarray) -> Project:
    h, w = mask.shape
    project = Project(canvas_resolution=(w, h))
    conductor = Conductor(mask=mask, voltage=1.0, position=(0.0, 0.0))
    project.conductors.append(conductor)
    return project


def test_full_pipeline_matches_reference():
    mask = load_alpha(str(ASSET))
    project = prepare_project(mask)
    base_min = float(min(project.canvas_resolution))
    project.streamlength_factor = 30.0 / base_min

    multiplier = 2
    supersample = 1.5
    scale = multiplier * supersample
    compute_w = int(round(project.canvas_resolution[0] * scale))
    compute_h = int(round(project.canvas_resolution[1] * scale))
    compute_min = min(compute_w, compute_h)

    _, (ex, ey) = compute_field_pde(project, multiplier=multiplier, supersample=supersample)

    rng = np.random.default_rng(0)
    seed_texture = rng.random((compute_h, compute_w)).astype(np.float32)

    pixel_streamlength = max(int(round(project.streamlength_factor * compute_min)), 1)
    ours_lic = compute_lic(
        ex,
        ey,
        pixel_streamlength,
        num_passes=1,
        texture=seed_texture,
        seed=None,
        boundaries="closed",
    )
    sigma_factor = 3.0 / 1024.0
    sigma_pixels = sigma_factor * compute_min
    ours_post = apply_gaussian_highpass(ours_lic, sigma_pixels)
    downsample_sigma = 0.6 * supersample
    render_shape = (
        project.canvas_resolution[1] * multiplier,
        project.canvas_resolution[0] * multiplier,
    )
    ours_down = downsample_lic(ours_post, render_shape, supersample, downsample_sigma)
    ours_min, ours_max = float(ours_down.min()), float(ours_down.max())
    ours_display = (ours_down - ours_min) / (ours_max - ours_min + 1e-10)

    mag = np.sqrt(ex**2 + ey**2)
    mag_max = float(np.max(mag)) if np.max(mag) > 1e-10 else 1e-10
    vx = (ex / mag_max).astype(np.float32)
    vy = (ey / mag_max).astype(np.float32)

    vfield = np.stack([vx, vy], axis=0)

    reference = compute_lic_with_postprocessing(
        vfield,
        sfield_in=seed_texture,
        streamlength=pixel_streamlength,
        seed_sfield=None,
        use_periodic_BCs=False,
        num_lic_passes=1,
        use_filter=True,
        filter_sigma=sigma_pixels,
        use_equalize=False,  # No longer using CLAHE
        backend="rust",
        run_in_parallel=True,
        verbose=False,
    )
    reference_down = downsample_lic(reference, render_shape, supersample, downsample_sigma)
    ref_min, ref_max = float(reference_down.min()), float(reference_down.max())
    reference_display = (reference_down - ref_min) / (ref_max - ref_min + 1e-10)

    diff = np.abs(ours_display - reference_display)
    mean_diff = float(diff.mean())
    max_diff = float(diff.max())

    assert (mean_diff < 1e-2) and (max_diff < 5e-2), (
        f"Pipelines diverge: mean={mean_diff:.4f} max={max_diff:.4f}"
    )
