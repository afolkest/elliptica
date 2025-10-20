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
