#!/usr/bin/env python3
"""Debug script to profile smear performance."""

import time
import torch
from flowcol.gpu import GPUContext
from flowcol.gpu.ops import gaussian_blur_gpu

# Simulate your 7k render with 2 thin strip conductors
render_size = 7000
print(f"Simulating 7k render ({render_size}x{render_size}):")
print()

# Create full LIC field
lic_field = torch.rand(render_size, render_size, device=GPUContext.device())
print(f"LIC field: {lic_field.shape}")

# Conductor 1: Thin vertical strip (7000 tall x 50 wide)
mask1 = torch.zeros(render_size, render_size, device=GPUContext.device())
mask1[:, 1000:1050] = 1.0  # 50px wide vertical strip

# Conductor 2: Thin horizontal strip (50 tall x 7000 wide)
mask2 = torch.zeros(render_size, render_size, device=GPUContext.device())
mask2[3000:3050, :] = 1.0  # 50px tall horizontal strip

print(f"Conductor 1 mask: {mask1.shape}, active pixels: {torch.sum(mask1 > 0.5).item()}")
print(f"Conductor 2 mask: {mask2.shape}, active pixels: {torch.sum(mask2 > 0.5).item()}")
print()

# Simulate smear processing
sigma_px = 10.0
kernel_size = max(3, (int(6 * sigma_px)) | 1)
print(f"Blur sigma: {sigma_px}px (kernel size: {kernel_size})")
print()

total_start = time.time()

for idx, mask in enumerate([mask1, mask2], 1):
    print(f"Processing conductor {idx}:")

    # Extract bounding box (what my optimization does)
    mask_bool = mask > 0.5
    mask_coords = torch.nonzero(mask_bool)

    y_min = mask_coords[:, 0].min().item()
    y_max = mask_coords[:, 0].max().item()
    x_min = mask_coords[:, 1].min().item()
    x_max = mask_coords[:, 1].max().item()

    # Add padding
    pad = int(3 * sigma_px) + 1
    y_min_pad = max(0, y_min - pad)
    y_max_pad = min(render_size, y_max + pad + 1)
    x_min_pad = max(0, x_min - pad)
    x_max_pad = min(render_size, x_max + pad + 1)

    region_h = y_max_pad - y_min_pad
    region_w = x_max_pad - x_min_pad

    print(f"  Bounding box: {region_w}x{region_h} pixels")
    print(f"  Coverage: {100*region_w*region_h/(render_size*render_size):.2f}% of full image")

    # Extract and blur region
    lic_region = lic_field[y_min_pad:y_max_pad, x_min_pad:x_max_pad]

    start = time.time()
    lic_blur_region = gaussian_blur_gpu(lic_region, sigma_px)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elapsed = time.time() - start

    print(f"  Blur time: {elapsed:.3f}s")
    print()

total_elapsed = time.time() - total_start
print(f"TOTAL time for 2 conductors: {total_elapsed:.3f}s")
print()
print("If you're seeing 20s, something else is the bottleneck!")
