#!/usr/bin/env python3

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "LineIntegralConvolutions" / "src"))

from flowcol.types import Project, Conductor
from flowcol.field import compute_field
from flowcol.render import compute_lic, apply_highpass_clahe
from flowcol.mask_utils import load_alpha
from vegtamr.lic import compute_lic_with_postprocessing


ASSET = ROOT / "assets" / "masks" / "disciples_shell.png"


def prepare_project(mask: np.ndarray) -> Project:
    h, w = mask.shape
    project = Project(canvas_resolution=(w, h), streamlength=30)
    conductor = Conductor(mask=mask, voltage=1.0, position=(0.0, 0.0))
    project.conductors.append(conductor)
    return project


def test_full_pipeline_matches_reference():
    mask = load_alpha(str(ASSET))
    project = prepare_project(mask)

    ex, ey = compute_field(project, multiplier=1)

    rng = np.random.default_rng(0)
    seed_texture = rng.random(project.canvas_resolution[::-1]).astype(np.float32)

    ours_lic = compute_lic(
        ex,
        ey,
        project.streamlength,
        num_passes=1,
        texture=seed_texture,
        seed=None,
        boundaries="closed",
    )
    ours_post = apply_highpass_clahe(
        ours_lic,
        sigma=3.0,
        clip_limit=0.01,
        kernel_rows=8,
        kernel_cols=8,
        num_bins=150,
    )
    ours_display = np.clip(ours_post, 0.0, 1.0)

    mag = np.sqrt(ex**2 + ey**2)
    mag_max = float(np.max(mag)) if np.max(mag) > 1e-10 else 1e-10
    vx = (ex / mag_max).astype(np.float32)
    vy = (ey / mag_max).astype(np.float32)

    vfield = np.stack([vx, vy], axis=0)

    reference = compute_lic_with_postprocessing(
        vfield,
        sfield_in=seed_texture,
        streamlength=project.streamlength,
        seed_sfield=None,
        use_periodic_BCs=False,
        num_lic_passes=1,
        use_filter=True,
        filter_sigma=3.0,
        use_equalize=True,
        backend="rust",
        run_in_parallel=True,
        verbose=False,
    )

    reference_display = np.clip(reference, 0.0, 1.0)

    diff = np.abs(ours_display - reference_display)
    mean_diff = float(diff.mean())
    max_diff = float(diff.max())

    assert (mean_diff < 1e-2) and (max_diff < 5e-2), (
        f"Pipelines diverge: mean={mean_diff:.4f} max={max_diff:.4f}"
    )
