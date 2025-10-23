#!/usr/bin/env python3
"""
Attempt to reproduce the gauss_law_morph orthodox saint render (streamlines only).

This script relies on FlowCol primitives and mirrors the pipeline:
1. Load the conductor shell mask, scaled to the requested canvas size.
2. Solve the electrostatic field with optional supersampling.
3. Run LIC with two passes and streamlength ≈120px (sl30).
4. Apply a single Gaussian high-pass (fs20 equivalent).
5. Downsample to the target resolution and map through twilight_shifted.

Metal styling and interior palette overrides are intentionally omitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowcol.types import Project, Conductor
from flowcol.field import compute_field
from flowcol.render import compute_lic, apply_gaussian_highpass, downsample_lic, _normalize_unit  # reuse normalization helper

def load_mask(path: Path) -> np.ndarray:
    """Load mask from RGBA alpha channel."""
    img = Image.open(path).convert("RGBA")
    alpha = np.array(img)[..., 3].astype(np.float32) / 255.0
    return alpha


def place_mask(mask: np.ndarray, canvas_size: Tuple[int, int], width_frac: float) -> Tuple[np.ndarray, Tuple[float, float]]:
    h, w = mask.shape
    canvas_w, canvas_h = canvas_size
    target_w = int(round(canvas_w * width_frac))
    scale = target_w / w
    target_h = int(round(h * scale))
    resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(
            (target_w, target_h), Image.Resampling.BILINEAR
        )
    ).astype(np.float32) / 255.0
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    x0 = (canvas_w - target_w) // 2
    y0 = (canvas_h - target_h) // 2
    canvas[y0:y0 + target_h, x0:x0 + target_w] = resized
    return canvas, (float(x0), float(y0))


def generate_white_noise(shape: Tuple[int, int], seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape).astype(np.float32)


def generate_blue_noise(shape: Tuple[int, int], seed: int | None = None, oversample: float = 1.5) -> np.ndarray:
    """Approximate blue noise generator mirroring gauss_law_morph."""
    rng = np.random.default_rng(seed)
    h, w = shape
    if oversample > 1.0:
        high_h = int(round(h * oversample))
        high_w = int(round(w * oversample))
    else:
        high_h, high_w = h, w

    noise = rng.random((high_h, high_w)).astype(np.float32)
    blue = noise.copy()
    for sigma in (8, 4, 2):
        smooth = gaussian_filter(blue, sigma=sigma)
        blue = blue - 0.5 * smooth

    blue = blue - blue.mean()
    blue = (blue - blue.min()) / (blue.max() - blue.min() + 1e-10)
    blue = 0.9 * blue + 0.1 * rng.random((high_h, high_w))
    blue = (blue - blue.min()) / (blue.max() - blue.min() + 1e-10)

    if (high_h, high_w) != (h, w):
        blue_img = Image.fromarray((blue * 255).astype(np.uint8), mode="L")
        blue = np.array(
            blue_img.resize((w, h), Image.Resampling.BILINEAR)
        ).astype(np.float32) / 255.0
    return blue.astype(np.float32)


def run_pipeline(
    mask_path: Path,
    interior_path: Path | None,
    output_path: Path,
    resolution: int,
    streamlength_physical: float,
    filter_sigma_factor: float,
    noise_type: str,
    seed: int | None,
    metal_out: Path | None = None,
) -> None:
    canvas_size = (resolution, resolution)
    # Prepare mask and conductor
    shell = load_mask(mask_path)
    mask_canvas, position = place_mask(shell, canvas_size, width_frac=0.8)

    project = Project(canvas_resolution=canvas_size)
    conductor = Conductor(mask=mask_canvas, voltage=10.0, position=position)
    project.conductors.append(conductor)

    ex, ey = compute_field(project)
    compute_h, compute_w = ex.shape
    compute_min = min(compute_h, compute_w)
    extent = 2.5  # gauss_law_morph physical half-width
    streamlength_px = int(round(streamlength_physical * compute_min / (2.0 * extent)))
    streamlength_px = max(streamlength_px, 1)
    project.streamlength_factor = streamlength_px / compute_min

    if noise_type == "white":
        texture = generate_white_noise((compute_h, compute_w), seed)
    elif noise_type == "blue":
        texture = generate_blue_noise((compute_h, compute_w), seed)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    lic = compute_lic(
        ex,
        ey,
        streamlength=streamlength_px,
        num_passes=2,
        texture=texture,
        seed=None,
        noise_sigma=0.0,  # Use raw white noise like gauss_law_morph (no pre-filtering)
    )

    raw_lic = lic.copy()

    if filter_sigma_factor > 0.0:
        sigma_px = filter_sigma_factor * compute_min
        lic = apply_gaussian_highpass(lic, sigma_px)

    lic_down = downsample_lic(lic, (resolution, resolution), supersample=1.0, sigma=0.0)

    # Use percentile normalization like gauss_law_morph (colorschemes.py:478-485)
    vmin = float(np.percentile(lic_down, 0.5))
    vmax = float(np.percentile(lic_down, 99.5))
    if vmax > vmin:
        norm = np.clip((lic_down - vmin) / (vmax - vmin + 1e-10), 0.0, 1.0)
    else:
        norm = _normalize_unit(lic_down)
    cmap = plt.get_cmap("twilight_shifted")
    rgb = cmap(norm)[:, :, :3]
    rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    Image.fromarray(rgb_uint8, mode="RGB").save(output_path)
    print(f"Saved render to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, default=Path("../gauss_law_morph/raw_images_to_use/orthodox_saint_shell_thin.png"))
    parser.add_argument("--interior", type=Path, default=Path("../gauss_law_morph/raw_images_to_use/orthodox_saint_interior_plus_shell.png"))
    parser.add_argument("--output", type=Path, default=Path("renders/orthodox_saint_flowcol.png"))
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--streamlength", type=float, default=0.30, help="Physical streamlength (test preset uses 0.30).")
    parser.add_argument("--filter", type=float, default=0.02)
    parser.add_argument("--noise", choices=["white", "blue"], default="white")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_pipeline(
        mask_path=args.mask,
        interior_path=args.interior,
        output_path=args.output,
        resolution=args.resolution,
        streamlength_physical=args.streamlength,
        filter_sigma_factor=args.filter,
        noise_type=args.noise,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
