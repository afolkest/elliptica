"""Texture management for Dear PyGui rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Tuple
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from flowcol.ui.dpg.app import FlowColApp

try:
    import dearpygui.dearpygui as dpg  # type: ignore
except ImportError:
    dpg = None


def _mask_to_rgba(mask: np.ndarray, color: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert mask to RGBA float texture."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = color[:3]
    rgba[..., 3] = mask.astype(np.float32) * color[3]
    return rgba.reshape(-1)


def _image_to_texture_data(img: Image.Image) -> Tuple[int, int, np.ndarray]:
    """Convert PIL image to float RGBA data."""
    img = img.convert("RGBA")
    width, height = img.size
    rgba = np.asarray(img, dtype=np.float32) / 255.0
    return width, height, rgba.reshape(-1)


def _rgb_array_to_texture_data(rgb: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """Convert RGB uint8 array to float RGBA data (fast path, no PIL roundtrip).

    Args:
        rgb: RGB array (H, W, 3) uint8

    Returns:
        (width, height, rgba_flat) where rgba_flat is flattened float32 RGBA
    """
    height, width = rgb.shape[:2]

    # Add alpha channel (full opacity) - this is much faster than PIL convert
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255

    # Convert to float32 in [0, 1] and flatten
    rgba_float = rgba.astype(np.float32) / 255.0
    return width, height, rgba_float.reshape(-1)


class TextureManager:
    """Manages DearPyGui textures for conductor overlays and rendered images."""

    def __init__(self, app: "FlowColApp"):
        self.app = app
        self.texture_registry_id: Optional[int] = None
        self.colormap_registry_id: Optional[int] = None
        self.palette_colormaps: Dict[str, int] = {}  # palette_name -> colormap_tag
        self.render_texture_id: Optional[int] = None
        self.render_texture_size: Optional[Tuple[int, int]] = None
        self.conductor_textures: Dict[int, int] = {}  # conductor_idx -> texture_id
        self.conductor_texture_shapes: Dict[int, Tuple[int, int]] = {}  # conductor_idx -> (height, width)

    def create_registries(self) -> None:
        """Create texture and colormap registries in DPG."""
        if dpg is None:
            return

        # Create texture registry for dynamic textures
        self.texture_registry_id = dpg.add_texture_registry()

        # Create colormap registry and convert our palettes to DPG colormaps
        from flowcol.render import _RUNTIME_PALETTES
        self.colormap_registry_id = dpg.add_colormap_registry()
        for palette_name, colors_normalized in _RUNTIME_PALETTES.items():
            # DPG expects colors as [R, G, B, A] with values 0-255
            colors_255 = [[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255] for c in colors_normalized]
            tag = f"colormap_{palette_name.replace(' ', '_').replace('&', 'and')}"
            dpg.add_colormap(colors_255, qualitative=False, tag=tag, parent=self.colormap_registry_id)
            self.palette_colormaps[palette_name] = tag

    def rebuild_colormaps(self) -> None:
        """Rebuild all colormaps after palette changes."""
        if dpg is None:
            return

        # Clear existing colormap registry
        if hasattr(self, 'colormap_registry_id') and dpg.does_item_exist(self.colormap_registry_id):
            dpg.delete_item(self.colormap_registry_id, children_only=True)

        # Rebuild colormaps from current runtime palettes
        from flowcol.render import _RUNTIME_PALETTES
        self.palette_colormaps.clear()

        for palette_name, colors_normalized in _RUNTIME_PALETTES.items():
            colors_255 = [[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255] for c in colors_normalized]
            tag = f"colormap_{palette_name.replace(' ', '_').replace('&', 'and')}"
            dpg.add_colormap(colors_255, qualitative=False, tag=tag, parent=self.colormap_registry_id)
            self.palette_colormaps[palette_name] = tag

    def ensure_conductor_texture(self, idx: int, mask: np.ndarray, conductor_colors: list) -> int:
        """Create or update conductor texture, returns texture ID.

        Args:
            idx: Conductor index
            mask: Conductor mask array (height, width)
            conductor_colors: List of RGBA colors for conductors

        Returns:
            DPG texture ID
        """
        assert dpg is not None and self.texture_registry_id is not None

        tex_id = self.conductor_textures.get(idx)
        width = mask.shape[1]
        height = mask.shape[0]
        existing_shape = self.conductor_texture_shapes.get(idx)

        # Check if texture needs recreation (doesn't exist or size changed)
        if tex_id is not None:
            exists = dpg.does_item_exist(tex_id)
            if not exists or existing_shape != (height, width):
                if exists:
                    dpg.delete_item(tex_id)
                tex_id = None
                self.conductor_textures.pop(idx, None)

        # Only convert and upload if texture doesn't exist (avoids 45 GB/sec bandwidth waste)
        if tex_id is None:
            # Convert mask to RGBA texture data
            rgba_flat = _mask_to_rgba(mask, conductor_colors[idx % len(conductor_colors)])
            tex_id = dpg.add_dynamic_texture(width, height, rgba_flat, parent=self.texture_registry_id)
            self.conductor_textures[idx] = tex_id

        self.conductor_texture_shapes[idx] = (height, width)
        return tex_id

    def refresh_render_texture(self) -> None:
        """Update render texture with postprocessed image from cache."""
        from PIL import Image

        if dpg is None or self.texture_registry_id is None:
            return

        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None or cache.result is None:
                # Fallback to grayscale if no render
                from flowcol.render import array_to_pil
                arr = np.zeros((32, 32), dtype=np.float32)
                pil_img = array_to_pil(arr, use_color=False)
                # Convert PIL image to texture data
                width, height, data = _image_to_texture_data(pil_img)
            else:
                # Apply postprocessing at FULL render resolution
                # DearPyGUI will scale it for display
                from flowcol.gpu.postprocess import apply_full_postprocess_hybrid

                clip_percent = self.app.state.display_settings.clip_percent

                # Only reuse cached percentiles if they were computed for the same clip%.
                lic_percentiles = None
                if cache.lic_percentiles is not None:
                    cached_clip = cache.lic_percentiles_clip_percent
                    if cached_clip is not None and abs(cached_clip - clip_percent) < 0.01:
                        lic_percentiles = cache.lic_percentiles

                final_rgb, used_percentiles = apply_full_postprocess_hybrid(
                    scalar_array=cache.result.array,  # Full resolution!
                    conductor_masks=cache.conductor_masks,  # Full resolution!
                    interior_masks=cache.interior_masks,  # Full resolution!
                    conductor_color_settings=self.app.state.conductor_color_settings,
                    conductors=self.app.state.project.conductors,
                    render_shape=cache.result.array.shape,  # Full resolution shape
                    canvas_resolution=self.app.state.project.canvas_resolution,
                    clip_percent=clip_percent,
                    brightness=self.app.state.display_settings.brightness,
                    contrast=self.app.state.display_settings.contrast,
                    gamma=self.app.state.display_settings.gamma,
                    color_enabled=self.app.state.display_settings.color_enabled,
                    palette=self.app.state.display_settings.palette,
                    lic_percentiles=lic_percentiles,
                    use_gpu=True,
                    scalar_tensor=cache.result_gpu,  # Use full-res GPU tensor if available
                    conductor_masks_gpu=cache.conductor_masks_gpu,  # Use cached GPU masks (no repeated transfers!)
                    interior_masks_gpu=cache.interior_masks_gpu,  # Use cached GPU masks
                )

                # Update cache so future refreshes reuse the correct clip percent bounds.
                cache.lic_percentiles = used_percentiles
                cache.lic_percentiles_clip_percent = clip_percent

                # Fast path: convert RGB→RGBA directly (no PIL roundtrip!)
                width, height, data = _rgb_array_to_texture_data(final_rgb)

        # Create or update texture
        if self.render_texture_id is None or self.render_texture_size != (width, height):
            if self.render_texture_id is not None:
                dpg.delete_item(self.render_texture_id)
            self.render_texture_id = dpg.add_dynamic_texture(width, height, data, parent=self.texture_registry_id)
            self.render_texture_size = (width, height)
        else:
            dpg.set_value(self.render_texture_id, data)

    def clear_conductor_texture(self, idx: int) -> None:
        """Clear cached conductor texture (forces recreation on next draw)."""
        self.conductor_textures.pop(idx, None)
        self.conductor_texture_shapes.pop(idx, None)

    def clear_all_conductor_textures(self) -> None:
        """Clear all conductor textures."""
        self.conductor_textures.clear()
        self.conductor_texture_shapes.clear()
