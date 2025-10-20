import numpy as np
import pygame
from PIL import Image
from flowcol.types import Project
from flowcol.lic import convolve, get_cosine_kernel


def render_lic(ex, ey, project: Project):
    """Render LIC visualization as pygame surface."""
    field_h, field_w = ex.shape

    texture = np.random.default_rng(0).random((field_h, field_w)).astype(np.float32)

    mag = np.sqrt(ex**2 + ey**2)
    mag_max = max(mag.max(), 1e-10)
    vx = (ex / mag_max).astype(np.float32)
    vy = (ey / mag_max).astype(np.float32)

    kernel = get_cosine_kernel(project.streamlength).astype(np.float32)
    lic_result = convolve(texture, vx, vy, kernel, iterations=1)

    img = (np.clip(lic_result, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode='L').convert('RGB')
    return pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode)
