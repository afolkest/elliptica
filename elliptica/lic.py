import numpy as np
import brylic
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
