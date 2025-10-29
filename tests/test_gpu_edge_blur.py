"""Quick test of GPU edge blur performance."""

import time
import numpy as np
import torch

from flowcol.gpu import GPUContext
from flowcol.gpu.edge_blur import apply_anisotropic_edge_blur_gpu
from flowcol.postprocess.blur import apply_anisotropic_edge_blur

# Check GPU availability
print(f"GPU (MPS) available: {GPUContext.is_available()}")
if not GPUContext.is_available():
    print("⚠️  No GPU available, skipping test")
    exit(0)

# Create test data
H, W = 760, 960
print(f"\nTest dimensions: {H}x{W}")

# Generate synthetic LIC and field
np.random.seed(42)
lic_array = np.random.rand(H, W).astype(np.float32)
ex = np.random.randn(H, W).astype(np.float32)
ey = np.random.randn(H, W).astype(np.float32)

# Create a simple conductor mask (circle in center)
y, x = np.mgrid[:H, :W]
center_y, center_x = H // 2, W // 2
radius = 100
dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
mask = (dist < radius).astype(np.float32)
conductor_masks = [mask]

# Test parameters
sigma = 3.0
falloff = 20.0
strength = 1.0

print(f"\nTest parameters:")
print(f"  sigma: {sigma}")
print(f"  falloff: {falloff}")
print(f"  strength: {strength}")

# CPU test
print("\n🐌 Testing CPU edge blur...")
start = time.time()
cpu_result = apply_anisotropic_edge_blur(
    lic_array, ex, ey, conductor_masks, sigma, falloff, strength
)
cpu_time = time.time() - start
print(f"🐌 CPU edge blur: {cpu_time*1000:.0f}ms")

# GPU test
print("\n🚀 Testing GPU edge blur...")
lic_gpu = GPUContext.to_gpu(lic_array)
ex_gpu = GPUContext.to_gpu(ex)
ey_gpu = GPUContext.to_gpu(ey)

start = time.time()
gpu_result_tensor = apply_anisotropic_edge_blur_gpu(
    lic_gpu, ex_gpu, ey_gpu, conductor_masks, sigma, falloff, strength
)
torch.mps.synchronize()  # Wait for GPU to finish
gpu_time = time.time() - start
print(f"🚀 GPU edge blur: {gpu_time*1000:.0f}ms")

# Download result
gpu_result = GPUContext.to_cpu(gpu_result_tensor)

# Compare results
diff = np.abs(cpu_result - gpu_result)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
print(f"\n📊 Results comparison:")
print(f"  Max difference: {max_diff:.6f}")
print(f"  Mean difference: {mean_diff:.6f}")
print(f"  Speedup: {cpu_time/gpu_time:.1f}x")

if max_diff < 0.01:
    print("✅ GPU and CPU results match!")
else:
    print("⚠️  GPU and CPU results differ significantly")

print(f"\n🎯 Final timing:")
print(f"  CPU: {cpu_time*1000:.0f}ms")
print(f"  GPU: {gpu_time*1000:.0f}ms")
print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
