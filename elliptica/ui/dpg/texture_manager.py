"""Texture management for Dear PyGui rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Tuple
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

from elliptica.render import list_palette_colormap_colors

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
    """Manages DearPyGui textures for boundary overlays and rendered images."""

    def __init__(self, app: "EllipticaApp"):
        self.app = app
        self.texture_registry_id: Optional[int] = None
        self.colormap_registry_id: Optional[int] = None
        self.palette_colormaps: Dict[str, int] = {}  # palette_name -> colormap_tag
        self.grayscale_colormap_tag: Optional[str] = None
        self.render_texture_id: Optional[int] = None
        self.render_texture_size: Optional[Tuple[int, int]] = None
        self.boundary_textures: Dict[int, int] = {}  # boundary_idx -> texture_id
        self.boundary_texture_shapes: Dict[int, Tuple[int, int]] = {}  # boundary_idx -> (height, width)

    def create_registries(self) -> None:
        """Create texture and colormap registries in DPG."""
        if dpg is None:
            return

        # Create texture registry for dynamic textures
        self.texture_registry_id = dpg.add_texture_registry()

        # Create colormap registry and convert our palettes to DPG colormaps
        self.colormap_registry_id = dpg.add_colormap_registry()
        for palette_name, colors_normalized in list_palette_colormap_colors().items():
            # DPG expects colors as [R, G, B, A] with values 0-255
            colors_255 = [[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255] for c in colors_normalized]
            tag = f"colormap_{palette_name.replace(' ', '_').replace('&', 'and')}"
            dpg.add_colormap(colors_255, qualitative=False, tag=tag, parent=self.colormap_registry_id)
            self.palette_colormaps[palette_name] = tag

        # Add grayscale colormap for preview use
        self.grayscale_colormap_tag = "colormap_grayscale"
        dpg.add_colormap(
            [[0, 0, 0, 255], [255, 255, 255, 255]],
            qualitative=False,
            tag=self.grayscale_colormap_tag,
            parent=self.colormap_registry_id,
        )

    def rebuild_colormaps(self) -> None:
        """Rebuild all colormaps after palette changes."""
        if dpg is None:
            return

        # Clear existing colormap registry
        if hasattr(self, 'colormap_registry_id') and dpg.does_item_exist(self.colormap_registry_id):
            dpg.delete_item(self.colormap_registry_id, children_only=True)

        # Rebuild colormaps from current runtime palettes
        self.palette_colormaps.clear()

        for palette_name, colors_normalized in list_palette_colormap_colors().items():
            colors_255 = [[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255] for c in colors_normalized]
            tag = f"colormap_{palette_name.replace(' ', '_').replace('&', 'and')}"
            dpg.add_colormap(colors_255, qualitative=False, tag=tag, parent=self.colormap_registry_id)
            self.palette_colormaps[palette_name] = tag

        self.grayscale_colormap_tag = "colormap_grayscale"
        dpg.add_colormap(
            [[0, 0, 0, 255], [255, 255, 255, 255]],
            qualitative=False,
            tag=self.grayscale_colormap_tag,
            parent=self.colormap_registry_id,
        )

    def ensure_boundary_texture(self, idx: int, mask: np.ndarray, boundary_colors: list) -> int:
        """Create or update boundary texture, returns texture ID.

        Args:
            idx: Boundary index
            mask: Boundary mask array (height, width)
            boundary_colors: List of RGBA colors for boundaries

        Returns:
            DPG texture ID
        """
        assert dpg is not None and self.texture_registry_id is not None

        tex_id = self.boundary_textures.get(idx)
        width = mask.shape[1]
        height = mask.shape[0]
        existing_shape = self.boundary_texture_shapes.get(idx)

        # Check if texture needs recreation (doesn't exist or size changed)
        if tex_id is not None:
            exists = dpg.does_item_exist(tex_id)
            if not exists or existing_shape != (height, width):
                if exists:
                    dpg.delete_item(tex_id)
                tex_id = None
                self.boundary_textures.pop(idx, None)

        # Only convert and upload if texture doesn't exist (avoids 45 GB/sec bandwidth waste)
        if tex_id is None:
            # Convert mask to RGBA texture data
            rgba_flat = _mask_to_rgba(mask, boundary_colors[idx % len(boundary_colors)])
            tex_id = dpg.add_dynamic_texture(width, height, rgba_flat, parent=self.texture_registry_id)
            self.boundary_textures[idx] = tex_id

        self.boundary_texture_shapes[idx] = (height, width)
        return tex_id

    def refresh_render_texture(self) -> None:
        """Update render texture with postprocessed image from cache."""
        from PIL import Image
        from elliptica.expr import ExprError

        if dpg is None or self.texture_registry_id is None:
            return

        width = None
        height = None
        data = None
        expr_error_msg = None

        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None or cache.result is None:
                # Fallback to grayscale if no render
                from elliptica.render import array_to_pil
                arr = np.zeros((32, 32), dtype=np.float32)
                pil_img = array_to_pil(arr, use_color=False)
                # Convert PIL image to texture data
                width, height, data = _image_to_texture_data(pil_img)
            else:
                # Apply postprocessing at FULL render resolution
                # DearPyGUI will scale it for display
                from elliptica.gpu.postprocess import apply_full_postprocess_hybrid

                clip_low = self.app.state.display_settings.clip_low_percent
                clip_high = self.app.state.display_settings.clip_high_percent

                # Only reuse cached percentiles if they were computed for the same clip%.
                lic_percentiles = None
                if cache.lic_percentiles is not None:
                    cached_clip = cache.lic_percentiles_clip_range
                    if cached_clip is not None:
                        cached_low, cached_high = cached_clip
                        if abs(cached_low - clip_low) < 0.01 and abs(cached_high - clip_high) < 0.01:
                            lic_percentiles = cache.lic_percentiles

                # Get or compute resized solution tensors (cached to avoid non-determinism)
                solution_gpu = None
                if cache.solution_gpu and cache.result_gpu is not None:
                    import torch
                    target_shape = cache.result_gpu.shape
                    if (cache.solution_gpu_resized is not None and
                        cache.solution_gpu_lic_shape == target_shape):
                        solution_gpu = cache.solution_gpu_resized
                    else:
                        resized = {}
                        for name, tensor in cache.solution_gpu.items():
                            if tensor.shape == target_shape:
                                resized[name] = tensor
                            else:
                                tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
                                r = torch.nn.functional.interpolate(
                                    tensor_4d, size=target_shape, mode='bilinear', align_corners=False
                                )
                                resized[name] = r.squeeze(0).squeeze(0)
                        cache.solution_gpu_resized = resized
                        cache.solution_gpu_lic_shape = target_shape
                        solution_gpu = resized

                try:
                    final_rgb, used_percentiles = apply_full_postprocess_hybrid(
                        scalar_array=cache.result.array,  # Full resolution!
                        boundary_masks=cache.boundary_masks,  # Full resolution!
                        interior_masks=cache.interior_masks,  # Full resolution!
                        boundary_color_settings=self.app.state.boundary_color_settings,
                        boundaries=self.app.state.project.boundary_objects,
                        render_shape=cache.result.array.shape,  # Full resolution shape
                        canvas_resolution=self.app.state.project.canvas_resolution,
                        clip_low_percent=clip_low,
                        clip_high_percent=clip_high,
                        brightness=self.app.state.display_settings.brightness,
                        contrast=self.app.state.display_settings.contrast,
                        gamma=self.app.state.display_settings.gamma,
                        color_enabled=self.app.state.display_settings.color_enabled,
                        palette=self.app.state.display_settings.palette,
                        lic_percentiles=lic_percentiles,
                        use_gpu=True,
                        scalar_tensor=cache.result_gpu,  # Use full-res GPU tensor if available
                        boundary_masks_gpu=cache.boundary_masks_gpu,  # Use cached GPU masks (no repeated transfers!)
                        interior_masks_gpu=cache.interior_masks_gpu,  # Use cached GPU masks
                        color_config=self.app.state.color_config,  # Expression-based coloring (if set)
                        ex_tensor=cache.ex_gpu,  # Field components for ColorConfig mag binding
                        ey_tensor=cache.ey_gpu,
                        lightness_expr=self.app.state.display_settings.lightness_expr,
                        solution_gpu=solution_gpu,  # PDE solution fields resized to LIC resolution
                        saturation=self.app.state.display_settings.saturation,
                    )

                    # Update cache so future refreshes reuse the correct clip bounds.
                    cache.lic_percentiles = used_percentiles
                    cache.lic_percentiles_clip_range = (clip_low, clip_high)

                    # Fast path: convert RGB→RGBA directly (no PIL roundtrip!)
                    width, height, data = _rgb_array_to_texture_data(final_rgb)

                except ExprError as e:
                    # Expression evaluation error - keep previous display, show error
                    expr_error_msg = str(e)

        # Update expression error display if postprocessing panel has one
        if hasattr(self.app, 'postprocess_panel') and self.app.postprocess_panel is not None:
            if dpg.does_item_exist("expr_error_text"):
                if expr_error_msg:
                    dpg.set_value("expr_error_text", f"Error: {expr_error_msg}")
                # Don't clear error on success - let the panel manage that

        # If there was an error, don't update the texture (keep previous valid display)
        if data is None:
            return

        # Create or update texture
        if self.render_texture_id is None or self.render_texture_size != (width, height):
            if self.render_texture_id is not None:
                dpg.delete_item(self.render_texture_id)
            self.render_texture_id = dpg.add_dynamic_texture(width, height, data, parent=self.texture_registry_id)
            self.render_texture_size = (width, height)
        else:
            dpg.set_value(self.render_texture_id, data)

    def update_texture_from_rgb(self, rgb: np.ndarray) -> None:
        """Update render texture from precomputed RGB array.

        Used by async postprocessing to update texture on main thread.

        Args:
            rgb: RGB array (H, W, 3) uint8
        """
        if dpg is None or self.texture_registry_id is None:
            return

        width, height, data = _rgb_array_to_texture_data(rgb)

        # Create or update texture
        if self.render_texture_id is None or self.render_texture_size != (width, height):
            if self.render_texture_id is not None:
                dpg.delete_item(self.render_texture_id)
            self.render_texture_id = dpg.add_dynamic_texture(width, height, data, parent=self.texture_registry_id)
            self.render_texture_size = (width, height)
        else:
            dpg.set_value(self.render_texture_id, data)

    def clear_boundary_texture(self, idx: int) -> None:
        """Clear cached boundary texture (forces recreation on next draw)."""
        self.boundary_textures.pop(idx, None)
        self.boundary_texture_shapes.pop(idx, None)

    def clear_all_boundary_textures(self) -> None:
        """Clear all boundary textures."""
        self.boundary_textures.clear()
        self.boundary_texture_shapes.clear()
