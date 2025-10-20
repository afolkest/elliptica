import numpy as np
import pygame
from PIL import Image
from pathlib import Path
from datetime import datetime
from flowcol.types import Project, RenderInfo
from flowcol.lic import convolve, get_cosine_kernel


def compute_lic(ex, ey, project: Project) -> np.ndarray:
    """Compute LIC visualization. Returns normalized grayscale array [0, 1]."""
    field_h, field_w = ex.shape

    texture = np.random.default_rng(0).random((field_h, field_w)).astype(np.float32)

    mag = np.sqrt(ex**2 + ey**2)
    mag_max = max(mag.max(), 1e-10)
    vx = (ex / mag_max).astype(np.float32)
    vy = (ey / mag_max).astype(np.float32)

    kernel = get_cosine_kernel(project.streamlength).astype(np.float32)
    lic_result = convolve(texture, vx, vy, kernel, iterations=1)

    return np.clip(lic_result, 0, 1)


def array_to_surface(arr: np.ndarray) -> pygame.Surface:
    """Convert grayscale array to pygame surface."""
    img = (arr * 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode='L').convert('RGB')
    return pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode)


def save_render(arr: np.ndarray, project: Project, multiplier: int) -> RenderInfo:
    """Save render to file and return RenderInfo."""
    renders_dir = Path("renders")
    renders_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lic_{multiplier}x_{timestamp}.png"
    filepath = renders_dir / filename

    img = (arr * 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(filepath)

    render_info = RenderInfo(multiplier=multiplier, filepath=str(filepath), timestamp=timestamp)
    project.renders.append(render_info)

    return render_info


def apply_contrast(arr: np.ndarray, factor: float) -> np.ndarray:
    """Apply contrast adjustment. factor: 0.5-2.0, 1.0 = no change."""
    mean = arr.mean()
    return np.clip((arr - mean) * factor + mean, 0, 1)


def apply_brightness(arr: np.ndarray, offset: float) -> np.ndarray:
    """Apply brightness offset. offset: -0.5 to 0.5."""
    return np.clip(arr + offset, 0, 1)


def invert(arr: np.ndarray) -> np.ndarray:
    """Invert grayscale values."""
    return 1.0 - arr


def apply_colormap(arr: np.ndarray, colormap_name: str) -> np.ndarray:
    """Apply colormap to grayscale. Returns RGB array [0, 1]."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap_name)
    rgb = cmap(arr)[:, :, :3]
    return rgb
