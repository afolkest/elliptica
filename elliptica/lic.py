import numpy as np
import brylic
from scipy.ndimage import gaussian_filter, zoom
from elliptica import defaults


def convolve(
    texture,
    vx,
    vy,
    kernel,
    iterations=1,
    boundaries="closed",
    mask=None,
    edge_gain_strength=None,
    edge_gain_power=None,
    domain_edge_gain_strength=None,
    domain_edge_gain_power=None,
):
    """
    Convolve a texture with a vector field using a kernel to produce a LIC image.

    Uses bryLIC with tiling for improved performance and mask support.

    Args:
        texture: The texture to convolve. Can be grayscale or color. 2d or 3d numpy array.
        vx: The x component of the vector field. 2d numpy array.
        vy: The y component of the vector field. 2d numpy array.
        kernel: The kernel to use for the convolution. 1d numpy array.
        iterations: The number of iterations to use for the convolution. Default is 1.
        boundaries: Boundary condition ("closed" or "periodic"). Default is "closed".
        mask: Optional boolean mask to block streamlines (boundaries). None for no blocking.
        edge_gain_strength: Brightness boost near mask edges (0.0 = none, higher = brighter halos).
        edge_gain_power: Falloff curve sharpness for mask edge gain (higher = sharper edge).
        domain_edge_gain_strength: Brightness boost near domain edges (0.0 = none, higher = brighter halos).
        domain_edge_gain_power: Falloff curve sharpness for domain edge gain (higher = sharper edge).

    Returns:
        The convolved texture. 2d or 3d numpy array.
    """
    if edge_gain_strength is None:
        edge_gain_strength = defaults.DEFAULT_EDGE_GAIN_STRENGTH
    if edge_gain_power is None:
        edge_gain_power = defaults.DEFAULT_EDGE_GAIN_POWER
    if domain_edge_gain_strength is None:
        domain_edge_gain_strength = defaults.DEFAULT_DOMAIN_EDGE_GAIN_STRENGTH
    if domain_edge_gain_power is None:
        domain_edge_gain_power = defaults.DEFAULT_DOMAIN_EDGE_GAIN_POWER

    return brylic.tiled_convolve(
        texture,
        vx,
        vy,
        kernel=kernel,
        iterations=iterations,
        boundaries=boundaries,
        mask=mask,
        edge_gain_strength=edge_gain_strength,
        edge_gain_power=edge_gain_power,
        domain_edge_gain_strength=domain_edge_gain_strength,
        domain_edge_gain_power=domain_edge_gain_power,
        tile_shape=defaults.DEFAULT_TILE_SHAPE,
        num_threads=defaults.DEFAULT_NUM_THREADS,
    )

def get_cosine_kernel(streamlength=30):
    """Gives a cosine kernel for a given streamlength in pixel units."""
    positions = np.arange(1 - streamlength, streamlength, dtype=np.float32)
    return 0.5 * (1.0 + np.cos(np.pi * positions / streamlength))


def generate_noise(
    shape: tuple[int, int],
    seed: int | None,
    oversample: float = 1.0,
    lowpass_sigma: float = defaults.DEFAULT_NOISE_SIGMA,
) -> np.ndarray:
    height, width = shape
    rng = np.random.default_rng(seed)
    if oversample > 1.0:
        high_h = max(1, int(round(height * oversample)))
        high_w = max(1, int(round(width * oversample)))
        base = rng.random((high_h, high_w)).astype(np.float32)
    else:
        base = rng.random((height, width)).astype(np.float32)

    if lowpass_sigma > 0.0:
        base = gaussian_filter(base, sigma=lowpass_sigma)

    if oversample > 1.0:
        scale_y = height / base.shape[0]
        scale_x = width / base.shape[1]
        base = zoom(base, (scale_y, scale_x), order=1)

    base_min = float(base.min())
    base_max = float(base.max())
    if base_max > base_min:
        base = (base - base_min) / (base_max - base_min)
    else:
        base = np.zeros_like(base)
    return base.astype(np.float32)


def compute_lic(
    ex: np.ndarray,
    ey: np.ndarray,
    streamlength: int,
    num_passes: int = 1,
    *,
    texture: np.ndarray | None = None,
    seed: int | None = 0,
    boundaries: str = "closed",
    noise_oversample: float = 1.5,
    noise_sigma: float = defaults.DEFAULT_NOISE_SIGMA,
    mask: np.ndarray | None = None,
    edge_gain_strength: float = 0.0,
    edge_gain_power: float = 2.0,
    domain_edge_gain_strength: float = 0.0,
    domain_edge_gain_power: float = 2.0,
) -> np.ndarray:
    """Compute LIC visualization. Returns array normalized to [-1, 1].

    Args:
        ex: X component of electric field
        ey: Y component of electric field
        streamlength: Streamline length in pixels
        num_passes: Number of LIC iterations
        texture: Optional input texture (generates white noise if None)
        seed: Random seed for noise generation
        boundaries: Boundary conditions ("closed" or "periodic")
        noise_oversample: Oversample factor for noise generation
        noise_sigma: Low-pass filter sigma for noise
        mask: Optional boolean mask to block streamlines (True = blocked)
        edge_gain_strength: Brightness boost near mask edges (0.0 = none)
        edge_gain_power: Falloff curve sharpness for mask edge gain
        domain_edge_gain_strength: Brightness boost near domain edges (0.0 = none)
        domain_edge_gain_power: Falloff curve sharpness for domain edge gain
    """
    field_h, field_w = ex.shape

    streamlength = max(int(streamlength), 1)

    # Scale noise_sigma with resolution (reference: 1024px on shorter side)
    scale_factor = min(field_h, field_w) / 1024.0
    scaled_sigma = noise_sigma * scale_factor

    if texture is None:
        texture = generate_noise(
            (field_h, field_w),
            seed,
            oversample=noise_oversample,
            lowpass_sigma=scaled_sigma,
        )
    else:
        texture = texture.astype(np.float32, copy=False)

    vx = ex.astype(np.float32, copy=False)
    vy = ey.astype(np.float32, copy=False)
    if not np.any(vx) and not np.any(vy):
        return np.zeros_like(ex, dtype=np.float32)

    kernel = get_cosine_kernel(streamlength).astype(np.float32)
    lic_result = convolve(
        texture,
        vx,
        vy,
        kernel,
        iterations=num_passes,
        boundaries=boundaries,
        mask=mask,
        edge_gain_strength=edge_gain_strength,
        edge_gain_power=edge_gain_power,
        domain_edge_gain_strength=domain_edge_gain_strength,
        domain_edge_gain_power=domain_edge_gain_power,
    )

    max_abs = np.max(np.abs(lic_result))
    if max_abs > 1e-12:
        lic_result = lic_result / max_abs

    return lic_result
