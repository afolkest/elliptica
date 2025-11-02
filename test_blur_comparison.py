#!/usr/bin/env python3
"""
Standalone script to compare different blur techniques for conductor interior.

Generates a side-by-side comparison panel showing:
1. Original LIC with white noise
2. Standard Gaussian blur (current approach)
3. Multi-scale decomposition
4. Bilateral filter
5. Frequency domain band-reject

No dependencies on GPU code - pure CPU/NumPy/SciPy.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from pathlib import Path


def generate_realistic_scene(shape=(512, 512), seed=42):
    """
    Generate realistic scenario:
    - Outside conductor: LIC texture (organized flowing lines)
    - Inside conductor: Raw white noise (E=0, no field to trace)

    This matches the actual physics: E=0 inside conductor.
    """
    rng = np.random.default_rng(seed)

    # Generate white noise (same noise everywhere initially)
    noise = rng.random(shape).astype(np.float32)

    # Normalize to [-1, 1]
    noise = (noise - 0.5) * 2.0

    # Create LIC texture ONLY OUTSIDE conductor by smearing noise along field lines
    # Inside conductor, field is zero, so noise stays unsmeared

    # Simulate LIC by smearing noise along diagonal direction (outside only)
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    # Diagonal line kernel
    for i in range(kernel_size):
        kernel[i, i] = 1.0
    kernel /= kernel.sum()

    # Apply directional smear to get LIC
    lic_texture = fftconvolve(noise, kernel, mode='same')

    # Add some variation - mix of different scales
    for angle_deg in [45, -45, 0, 90]:
        angle_rad = np.radians(angle_deg)

        # Create oriented kernel
        k_size = 11
        k = np.zeros((k_size, k_size))
        center = k_size // 2
        for i in range(k_size):
            dx = i - center
            dy = int(dx * np.tan(angle_rad))
            if abs(dy) < center:
                k[center + dy, i] = 1.0
        if k.sum() > 0:
            k /= k.sum()
            lic_texture += 0.1 * fftconvolve(noise, k, mode='same')

    # Normalize to [-1, 1] range
    lic_texture = (lic_texture - lic_texture.mean()) / (lic_texture.std() + 1e-8)
    lic_texture = np.clip(lic_texture, -3, 3) / 3.0

    return lic_texture.astype(np.float32), noise.astype(np.float32)


def create_circular_mask(shape, center=None, radius=None):
    """Create a circular mask for conductor region."""
    if center is None:
        center = (shape[0] // 2, shape[1] // 2)
    if radius is None:
        radius = min(shape) // 3

    y, x = np.ogrid[:shape[0], :shape[1]]
    dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask = (dist <= radius).astype(np.float32)

    # Smooth edge
    mask = gaussian_filter(mask, sigma=2.0)
    return mask


# ============================================================================
# Blur Techniques
# ============================================================================

def gaussian_blur(image, sigma):
    """Standard Gaussian blur (current approach)."""
    return gaussian_filter(image, sigma=sigma)


def multiscale_blur(image, sigma, detail_preservation=0.3, decompose_sigma=3.0):
    """
    Multi-scale decomposition: smooth base + preserved detail.

    Args:
        image: Input image
        sigma: Blur strength for base layer
        detail_preservation: How much high-freq detail to keep (0-1)
        decompose_sigma: Scale for base/detail separation
    """
    # 1. Decompose into base + detail
    base = gaussian_filter(image, sigma=decompose_sigma)
    detail = image - base  # High-frequency microstructure

    # 2. Smooth base aggressively (kills mid-frequency blobs)
    base_smooth = gaussian_filter(base, sigma=sigma)

    # 3. Preserve but attenuate detail
    detail_preserved = detail * detail_preservation

    # 4. Recombine
    return base_smooth + detail_preserved


def bilateral_filter(image, sigma_spatial, sigma_intensity):
    """
    Edge-preserving bilateral filter.

    Simplified implementation - for production use cv2.bilateralFilter.
    """
    from scipy.ndimage import gaussian_filter

    # Simple approximation: use range filter
    # True bilateral is expensive, this is a cheap approximation
    blurred = gaussian_filter(image, sigma=sigma_spatial)

    # Preserve edges by mixing based on gradient
    grad_y, grad_x = np.gradient(image)
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    edge_weight = np.exp(-edge_strength**2 / (2 * sigma_intensity**2))

    # Mix original and blurred based on edge strength
    result = edge_weight * image + (1 - edge_weight) * blurred
    return result


def frequency_bandreject_blur(image, reject_band=(0.1, 0.3)):
    """
    Frequency domain band-reject filter.

    Kills mid-frequencies that look like blobby noise,
    preserves low (structure) and high (fine detail).

    Args:
        image: Input image
        reject_band: (low_freq, high_freq) normalized frequencies to reject
    """
    # FFT
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    # Create frequency grid
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]

    # Normalized distance from center
    dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
    max_dist = np.sqrt(crow**2 + ccol**2)
    norm_dist = dist / max_dist

    # Band-reject mask: 1 outside band, 0 inside band
    low_freq, high_freq = reject_band
    mask = np.ones_like(norm_dist)
    mask[(norm_dist >= low_freq) & (norm_dist <= high_freq)] = 0.0

    # Smooth transition (Gaussian rolloff)
    mask = gaussian_filter(mask, sigma=5.0)

    # Apply mask
    fft_filtered = fft_shift * mask

    # IFFT
    fft_ishift = np.fft.ifftshift(fft_filtered)
    result = np.fft.ifft2(fft_ishift)
    return np.real(result)


def apply_in_mask(image, blurred, mask):
    """Apply blur only inside mask region."""
    return image * (1 - mask) + blurred * mask


# ============================================================================
# Main Comparison
# ============================================================================

def main():
    print("Generating realistic scene...")
    print("  - Outside conductor: LIC (organized flowing lines)")
    print("  - Inside conductor: Raw white noise (E=0)")
    shape = (512, 512)
    lic_outside, noise_inside = generate_realistic_scene(shape, seed=42)

    print("Creating conductor mask...")
    mask = create_circular_mask(shape, radius=150)

    # Composite: LIC outside, white noise inside (realistic physics)
    # This is what the actual LIC output looks like
    base_image = lic_outside * (1 - mask) + noise_inside * mask

    # Blur parameters
    sigma = 5.0  # Standard blur strength (matches conductor.smear_sigma)

    print("Applying blur techniques to INTERIOR ONLY...")
    print(f"  (sigma={sigma}, blurring raw white noise inside conductor)")

    # 1. Original (no blur) - raw noise inside, LIC outside
    original = base_image.copy()

    # 2. Standard Gaussian blur (current approach)
    # Blur the INTERIOR white noise
    interior_blurred_gauss = gaussian_blur(noise_inside, sigma=sigma)
    gauss_result = lic_outside * (1 - mask) + interior_blurred_gauss * mask

    # 3. Multi-scale with different preservation levels
    # Apply to the white noise inside
    interior_ms_30 = multiscale_blur(noise_inside, sigma=sigma, detail_preservation=0.3)
    multiscale_30_result = lic_outside * (1 - mask) + interior_ms_30 * mask

    interior_ms_50 = multiscale_blur(noise_inside, sigma=sigma, detail_preservation=0.5)
    multiscale_50_result = lic_outside * (1 - mask) + interior_ms_50 * mask

    interior_ms_70 = multiscale_blur(noise_inside, sigma=sigma, detail_preservation=0.7)
    multiscale_70_result = lic_outside * (1 - mask) + interior_ms_70 * mask

    # 4. Bilateral filter on white noise
    interior_bilateral = bilateral_filter(noise_inside, sigma_spatial=sigma, sigma_intensity=0.2)
    bilateral_result = lic_outside * (1 - mask) + interior_bilateral * mask

    # 5. Frequency domain band-reject on white noise
    interior_freq = frequency_bandreject_blur(noise_inside, reject_band=(0.15, 0.35))
    freq_result = lic_outside * (1 - mask) + interior_freq * mask

    print("Creating comparison panel...")

    # Create figure with 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Conductor Interior Blur Comparison (sigma={sigma})\nInside: Raw white noise (E=0) | Outside: LIC texture', fontsize=14)

    images = [
        (original, "1. Original\n(raw noise inside)"),
        (gauss_result, "2. Gaussian Blur\n(current approach)"),
        (multiscale_30_result, "3. Multi-Scale\n(30% detail preserved)"),
        (multiscale_50_result, "4. Multi-Scale\n(50% detail preserved)"),
        (multiscale_70_result, "5. Multi-Scale\n(70% detail preserved)"),
        (bilateral_result, "6. Bilateral Filter\n(edge-preserving)"),
        (freq_result, "7. Frequency Band-Reject\n(kill mid-freq blobs)"),
        (lic_outside, "LIC Outside\n(organized texture)"),
        (noise_inside, "White Noise Inside\n(raw, unsmeared)"),
    ]

    for ax, (img, title) in zip(axes.flat, images):
        im = ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent / "blur_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to: {output_path}")

    # Also create zoomed-in comparison of the conductor edge
    print("Creating edge detail comparison...")

    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    fig2.suptitle(f'Edge Detail Comparison (conductor boundary)\nFocus on transition from organized LIC to processed noise', fontsize=13)

    # Crop to interesting region (conductor edge)
    cy, cx = shape[0] // 2, shape[1] // 2
    crop_size = 100
    y1, y2 = cy - crop_size, cy + crop_size
    x1, x2 = cx - crop_size, cx + crop_size

    crops = [
        (original[y1:y2, x1:x2], "1. Original\n(raw noise)"),
        (gauss_result[y1:y2, x1:x2], "2. Gaussian\n(current)"),
        (multiscale_30_result[y1:y2, x1:x2], "3. Multi-Scale 30%\n(subtle)"),
        (multiscale_50_result[y1:y2, x1:x2], "4. Multi-Scale 50%\n(moderate)"),
        (multiscale_70_result[y1:y2, x1:x2], "5. Multi-Scale 70%\n(high detail)"),
        (bilateral_result[y1:y2, x1:x2], "6. Bilateral\n(edge-aware)"),
        (freq_result[y1:y2, x1:x2], "7. Frequency\n(band-reject)"),
        (mask[y1:y2, x1:x2], "Mask\n(conductor region)"),
    ]

    for ax, (crop, title) in zip(axes2.flat, crops):
        ax.imshow(crop, cmap='gray', vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()

    output_path2 = Path(__file__).parent / "blur_comparison_detail.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved detail comparison to: {output_path2}")

    print("\nDone! Open the images to compare techniques.")
    print("\nThe Problem:")
    print("  - Inside conductor: E=0, so LIC has no field to trace → raw white noise")
    print("  - Outside conductor: Beautiful organized LIC flowing along field lines")
    print("  - Gaussian blur on white noise → ugly mid-frequency blobs")
    print("  - Visual clash between organized LIC and mushy blurred noise")
    print("\nRecommendations:")
    print("  - Multi-scale with 30-50% detail should look most natural")
    print("  - Preserves high-freq texture while smoothing the random character")
    print("  - Bilateral is also good but more expensive")
    print("  - Frequency method is interesting but might be overkill")


if __name__ == "__main__":
    main()
