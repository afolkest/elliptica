#!/usr/bin/env python3
"""Debug MPS synchronization overhead in the postprocessing pipeline.

This script measures where the 14.3s MPS sync delay is coming from.
"""

import time
import numpy as np
import torch
from flowcol.gpu import GPUContext
from flowcol.gpu.postprocess import apply_full_postprocess_gpu
from flowcol.types import Conductor

print("=" * 80)
print("MPS SYNCHRONIZATION DIAGNOSTIC")
print("=" * 80)

# Check if GPU is available
if not GPUContext.is_available():
    print("❌ GPU not available!")
    exit(1)

print(f"✓ GPU available: {torch.backends.mps.is_available()}")
print()

# Create a realistic 7k test case
render_shape = (7000, 7000)
print(f"Test resolution: {render_shape[0]}x{render_shape[1]} ({render_shape[0]*render_shape[1]/1e6:.1f}M pixels)")
print()

# Generate test data
print("Generating test data...")
scalar_array = np.random.rand(*render_shape).astype(np.float32)
scalar_tensor = GPUContext.to_gpu(scalar_array)
print(f"✓ Uploaded {scalar_array.nbytes / 1e6:.1f} MB to GPU")
print()

# Create minimal conductor setup (2 conductors with smear)
conductors = [
    Conductor(
        mask=np.zeros((100, 3000), dtype=np.float32),  # Thin strip
        voltage=1.0,
        smear_enabled=True,
        smear_sigma=0.0005,
        id=0,
    ),
    Conductor(
        mask=np.zeros((100, 3000), dtype=np.float32),  # Thin strip
        voltage=-1.0,
        smear_enabled=True,
        smear_sigma=0.0005,
        id=1,
    ),
]

# Create full-resolution masks
conductor_masks = []
for c in conductors:
    mask = np.zeros(render_shape, dtype=np.float32)
    # Place conductor in middle of image
    y_start = render_shape[0] // 2 - c.mask.shape[0] // 2
    x_start = render_shape[1] // 2 - c.mask.shape[1] // 2
    mask[y_start:y_start + c.mask.shape[0], x_start:x_start + c.mask.shape[1]] = 1.0
    conductor_masks.append(mask)

interior_masks = [None, None]

print("=" * 80)
print("TIMING MPS SYNCHRONIZATION POINTS")
print("=" * 80)
print()

# Test 1: Baseline GPU operation (no sync)
print("Test 1: Baseline GPU operations (no explicit sync)")
print("-" * 80)
start = time.time()
result = scalar_tensor * 2.0 + 1.0
result = torch.clamp(result, 0, 1)
elapsed = time.time() - start
print(f"GPU math operations: {elapsed*1000:.1f} ms (async, no sync)")
print()

# Test 2: Explicit MPS sync
print("Test 2: Explicit MPS synchronize()")
print("-" * 80)
start = time.time()
if torch.backends.mps.is_available():
    torch.mps.synchronize()
elapsed = time.time() - start
print(f"torch.mps.synchronize(): {elapsed*1000:.1f} ms")
print()

# Test 3: GPU→CPU transfer (implicit sync)
print("Test 3: GPU→CPU transfer (implicit sync)")
print("-" * 80)
start = time.time()
cpu_array = GPUContext.to_cpu(scalar_tensor)
elapsed = time.time() - start
print(f"to_cpu() transfer: {elapsed*1000:.1f} ms")
print(f"Transfer rate: {scalar_array.nbytes / elapsed / 1e9:.2f} GB/s")
print()

# Test 4: Full postprocessing pipeline WITHOUT pre-cached percentiles
print("Test 4: Full postprocessing pipeline (NO pre-cached percentiles)")
print("-" * 80)

start_total_noprecache = time.time()

rgb_tensor_noprecache = apply_full_postprocess_gpu(
    scalar_tensor=scalar_tensor,
    conductor_masks_cpu=conductor_masks,
    interior_masks_cpu=interior_masks,
    conductor_color_settings={},
    conductors=conductors,
    render_shape=render_shape,
    canvas_resolution=(1024, 1024),
    clip_percent=2.0,
    brightness=0.0,
    contrast=1.0,
    gamma=1.0,
    color_enabled=True,
    palette="viridis",
    lic_percentiles=None,  # Force percentile computation
)

if torch.backends.mps.is_available():
    torch.mps.synchronize()

elapsed_noprecache = time.time() - start_total_noprecache
print(f"TOTAL (no precache): {elapsed_noprecache*1000:.1f} ms")
print()

# Test 5: Full postprocessing pipeline WITH pre-cached percentiles
print("Test 5: Full postprocessing pipeline (WITH pre-cached percentiles)")
print("-" * 80)

# Precompute percentiles
vmin = float(np.percentile(scalar_array, 0.5))
vmax = float(np.percentile(scalar_array, 99.5))
lic_percentiles = (vmin, vmax)

start_total = time.time()

# Run postprocessing
start_gpu = time.time()
rgb_tensor = apply_full_postprocess_gpu(
    scalar_tensor=scalar_tensor,
    conductor_masks_cpu=conductor_masks,
    interior_masks_cpu=interior_masks,
    conductor_color_settings={},
    conductors=conductors,
    render_shape=render_shape,
    canvas_resolution=(1024, 1024),
    clip_percent=2.0,
    brightness=0.0,
    contrast=1.0,
    gamma=1.0,
    color_enabled=True,
    palette="viridis",
    lic_percentiles=lic_percentiles,  # Use pre-cached percentiles
)
gpu_elapsed = time.time() - start_gpu
print(f"GPU postprocessing: {gpu_elapsed*1000:.1f} ms")

# Convert to uint8 (still on GPU)
start_convert = time.time()
rgb_uint8_tensor = (rgb_tensor * 255.0).clamp(0, 255).to(torch.uint8)
convert_elapsed = time.time() - start_convert
print(f"uint8 conversion (GPU): {convert_elapsed*1000:.1f} ms")

# MPS synchronize
start_sync = time.time()
if torch.backends.mps.is_available():
    torch.mps.synchronize()
elif torch.cuda.is_available():
    torch.cuda.synchronize()
sync_elapsed = time.time() - start_sync
print(f"MPS synchronize: {sync_elapsed*1000:.1f} ms ⚠️")

# Download to CPU
start_download = time.time()
final_rgb = GPUContext.to_cpu(rgb_uint8_tensor)
download_elapsed = time.time() - start_download
print(f"CPU download: {download_elapsed*1000:.1f} ms")

total_elapsed = time.time() - start_total
print(f"TOTAL: {total_elapsed*1000:.1f} ms")
print()


# Analysis
print("=" * 80)
print("ANALYSIS: Pre-cached Percentiles Impact")
print("=" * 80)
print()

precache_savings = elapsed_noprecache - total_elapsed
print(f"Without pre-cached percentiles: {elapsed_noprecache*1000:.1f} ms")
print(f"With pre-cached percentiles:    {total_elapsed*1000:.1f} ms")
print(f"Savings from pre-caching:       {precache_savings*1000:.1f} ms ({100*precache_savings/elapsed_noprecache:.1f}%)")
print()

if sync_elapsed > 1.0:
    print(f"⚠️  ISSUE FOUND: MPS sync takes {sync_elapsed:.1f}s!")
    print()
    print("Possible causes:")
    print("1. MPS is finishing queued work (blur, percentile, etc.)")
    print("2. sync() is redundant - to_cpu() already syncs")
    print("3. Large operations buffered in MPS queue")
else:
    print(f"✓ MPS sync is reasonable: {sync_elapsed*1000:.1f}ms")

print()
print("=" * 80)
print("BREAKDOWN OF TOTAL TIME")
print("=" * 80)
breakdown = {
    "GPU postprocessing": gpu_elapsed,
    "uint8 conversion": convert_elapsed,
    "MPS synchronize": sync_elapsed,
    "CPU download": download_elapsed,
}

for name, t in sorted(breakdown.items(), key=lambda x: -x[1]):
    pct = 100 * t / total_elapsed
    print(f"{name:25s}: {t*1000:7.1f} ms ({pct:5.1f}%)")

print(f"{'TOTAL':25s}: {total_elapsed*1000:7.1f} ms")
print()

# Check if operations are being queued
print("=" * 80)
print("INVESTIGATING OPERATION QUEUING")
print("=" * 80)
print()

# Run individual operations and sync after each
operations = []

# Op 1: Percentile computation
start = time.time()
from flowcol.gpu.ops import percentile_clip_gpu
normalized, _, _ = percentile_clip_gpu(scalar_tensor, 2.0)
if torch.backends.mps.is_available():
    torch.mps.synchronize()
elapsed = time.time() - start
operations.append(("Percentile clip", elapsed))

# Op 2: Brightness/contrast/gamma
start = time.time()
from flowcol.gpu.ops import apply_contrast_gamma_gpu
adjusted = apply_contrast_gamma_gpu(normalized, 0.0, 1.0, 1.0)
if torch.backends.mps.is_available():
    torch.mps.synchronize()
elapsed = time.time() - start
operations.append(("Contrast/gamma", elapsed))

# Op 3: Palette LUT
start = time.time()
from flowcol.render import _get_palette_lut
from flowcol.gpu.ops import apply_palette_lut_gpu
lut = _get_palette_lut("viridis")
lut_tensor = GPUContext.to_gpu(lut)
rgb = apply_palette_lut_gpu(adjusted, lut_tensor)
if torch.backends.mps.is_available():
    torch.mps.synchronize()
elapsed = time.time() - start
operations.append(("Palette LUT", elapsed))

# Op 4: Gaussian blur (smear)
start = time.time()
from flowcol.gpu.ops import gaussian_blur_gpu
# Blur a 7000x100 strip (typical conductor region)
strip = scalar_tensor[:100, :]
blurred = gaussian_blur_gpu(strip, sigma=3.5)
if torch.backends.mps.is_available():
    torch.mps.synchronize()
elapsed = time.time() - start
operations.append(("Gaussian blur (100px strip)", elapsed))

print("Individual operations (with sync after each):")
for name, t in operations:
    print(f"  {name:30s}: {t*1000:7.1f} ms")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if sync_elapsed > 5.0:
    print("🔴 MPS sync is VERY slow - likely all GPU work is deferred until sync")
    print()
    print("Recommendations:")
    print("1. REMOVE explicit torch.mps.synchronize() call")
    print("   - to_cpu() already performs implicit synchronization")
    print("   - Explicit sync may be forcing early pipeline stall")
    print()
    print("2. Consider breaking up large operations if possible")
    print("3. Profile individual GPU operations to find bottlenecks")
elif sync_elapsed > 1.0:
    print("🟡 MPS sync is moderately slow")
    print("   - This may be unavoidable for large 7k renders")
    print("   - Consider removing explicit sync (to_cpu already syncs)")
else:
    print("🟢 MPS sync overhead is acceptable")

print()
print("✓ Diagnostic complete")
