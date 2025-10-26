"""Test GPU overhead with and without caching."""

import time
import numpy as np
import torch

from flowcol.gpu import GPUContext
from flowcol.gpu.edge_blur import apply_anisotropic_edge_blur_gpu

print(f"GPU (MPS) available: {GPUContext.is_available()}")
if not GPUContext.is_available():
    print("⚠️  No GPU available, skipping test")
    exit(0)

# Test at small resolution where overhead matters
H, W = 760, 960
print(f"\nTest dimensions: {H}x{W}")

# Generate test data
np.random.seed(42)
lic_array = np.random.rand(H, W).astype(np.float32)
ex = np.random.randn(H, W).astype(np.float32)
ey = np.random.randn(H, W).astype(np.float32)

# Create conductor mask
y, x = np.mgrid[:H, :W]
center_y, center_x = H // 2, W // 2
radius = 100
dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
mask = (dist < radius).astype(np.float32)
conductor_masks = [mask]

sigma = 3.0
falloff = 20.0
strength = 1.0

print("\n=== Test 1: WITH UPLOAD OVERHEAD (old approach) ===")
for i in range(3):
    start = time.time()

    # Upload every time (OLD approach - what my test script did)
    lic_gpu = GPUContext.to_gpu(lic_array)
    ex_gpu = GPUContext.to_gpu(ex)
    ey_gpu = GPUContext.to_gpu(ey)

    # Apply edge blur
    result_gpu = apply_anisotropic_edge_blur_gpu(
        lic_gpu, ex_gpu, ey_gpu, conductor_masks, sigma, falloff, strength
    )
    torch.mps.synchronize()

    # Download result
    result = GPUContext.to_cpu(result_gpu)

    elapsed = time.time() - start
    print(f"  Run {i+1}: {elapsed*1000:.0f}ms")

print("\n=== Test 2: WITHOUT UPLOAD OVERHEAD (cached approach) ===")
# Upload ONCE (NEW approach - what real app does)
lic_gpu = GPUContext.to_gpu(lic_array)
ex_gpu = GPUContext.to_gpu(ex)
ey_gpu = GPUContext.to_gpu(ey)

for i in range(3):
    start = time.time()

    # Use cached GPU tensors (no upload!)
    result_gpu = apply_anisotropic_edge_blur_gpu(
        lic_gpu, ex_gpu, ey_gpu, conductor_masks, sigma, falloff, strength
    )
    torch.mps.synchronize()

    # Download result
    result = GPUContext.to_cpu(result_gpu)

    elapsed = time.time() - start
    print(f"  Run {i+1}: {elapsed*1000:.0f}ms")

print("\n✅ With cached GPU tensors, there's NO upload overhead!")
print("   The real app keeps everything on GPU between slider adjustments.")
