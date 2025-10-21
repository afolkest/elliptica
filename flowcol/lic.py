import numpy as np
import rlic


def convolve(texture, vx, vy, kernel, iterations=1, boundaries="closed"):
    """
    Convolve a texture with a vector field using a kernel to produce a LIC image.

    Args:
        texture: The texture to convolve. Can be grayscale or color. 2d or 3d numpy array.
        vx: The x component of the vector field. 2d numpy array.
        vy: The y component of the vector field. 2d numpy array.
        kernel: The kernel to use for the convolution. 1d numpy array.
        iterations: The number of iterations to use for the convolution. Default is 1.

    Returns:
        The convolved texture. 2d or 3d numpy array.
    """
    return rlic.convolve(texture, vx, vy, kernel=kernel, iterations=iterations, boundaries=boundaries)

def get_cosine_kernel(streamlength=30):
    """Gives a cosine kernel for a given streamlength in pixel units."""
    positions = np.arange(1 - streamlength, streamlength, dtype=np.float32)
    return 0.5 * (1.0 + np.cos(np.pi * positions / streamlength))
