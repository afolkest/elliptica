"""Standalone palette explorer for FlowCol.

Loads a .flowcol project with cached render and provides real-time palette preview.
Allows rapid exploration of color schemes without re-rendering physics.

Usage:
    python tools/palette_explorer.py path/to/project.flowcol
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flowcol.serialization import load_project, load_render_cache
from flowcol.render import _RUNTIME_PALETTES, _get_palette_lut, add_palette
from flowcol.gpu import GPUContext
from flowcol.gpu.pipeline import build_base_rgb_gpu
from flowcol.lospec import fetch_random_palette

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    print("Error: dearpygui not installed. Install with: pip install dearpygui")
    sys.exit(1)


class PaletteExplorer:
    """Lightweight palette explorer for rapid color iteration."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        if not self.project_path.exists():
            raise FileNotFoundError(f"Project not found: {project_path}")

        # Load project
        self.state = load_project(str(self.project_path))

        # Load render cache from separate .cache file
        cache_path = self.project_path.with_suffix('.flowcol.cache')
        if not cache_path.exists():
            raise ValueError(
                f"No render cache found at {cache_path}. "
                f"Please render the project first using the main FlowCol app."
            )

        self.state.render_cache = load_render_cache(str(cache_path), self.state.project)
        if self.state.render_cache is None or self.state.render_cache.result is None:
            raise ValueError(f"Failed to load render cache from {cache_path}")

        # Extract LIC array
        self.lic_array = self.state.render_cache.result.array  # (H, W) float32 in [0, 1]
        self.height, self.width = self.lic_array.shape

        # Convert to GPU tensor for fast processing
        self.use_gpu = GPUContext.is_available()
        if self.use_gpu:
            self.lic_tensor = GPUContext.to_gpu(self.lic_array)
        else:
            self.lic_tensor = None

        # Current settings (use defaults if not set)
        from flowcol import defaults
        self.current_palette = self.state.display_settings.palette
        self.brightness = self.state.display_settings.brightness if self.state.display_settings.brightness != 1.0 else defaults.DEFAULT_BRIGHTNESS
        self.contrast = self.state.display_settings.contrast
        self.gamma = self.state.display_settings.gamma
        self.saturation = 1.0  # Saturation multiplier (1.0 = no change)

        # DPG resources
        self.texture_id: Optional[int] = None
        self.texture_registry_id: Optional[int] = None
        self.colormap_registry_id: Optional[int] = None
        self.palette_colormaps: dict[str, int] = {}
        self.palette_popup_id: Optional[int] = None

        # Temporary Lospec palette (not yet in library)
        self.lospec_palette: Optional[dict] = None  # {name, author, slug, colors}
        self.lospec_palette_lut: Optional[np.ndarray] = None
        self.max_lospec_colors: int = 8  # Max colors to sample from large palettes

        print(f"✓ Loaded {self.width}x{self.height} render from {self.project_path.name}")
        print(f"✓ GPU acceleration: {'enabled' if self.use_gpu else 'disabled'}")

    def _image_to_texture_data(self, img: Image.Image) -> tuple[int, int, np.ndarray]:
        """Convert PIL image to DPG texture format."""
        img = img.convert("RGBA")
        width, height = img.size
        rgba = np.asarray(img, dtype=np.float32) / 255.0
        return width, height, rgba.reshape(-1)

    def _build_palette_lut_hsv(self, colors: np.ndarray, size: int = 256) -> np.ndarray:
        """Build palette LUT using HSV interpolation to preserve saturation.

        Linear RGB interpolation creates muddy/gray midpoints.
        HSV interpolation maintains color viv ibrancy through transitions.

        Args:
            colors: Color stops (N, 3) in RGB [0, 1]
            size: LUT size (default 256)

        Returns:
            LUT array (size, 3) in RGB [0, 1]
        """
        import colorsys

        # Convert RGB stops to HSV
        hsv_stops = np.array([colorsys.rgb_to_hsv(*rgb) for rgb in colors])

        # Handle hue wraparound: interpolate via shortest path
        # e.g., red (0°) to red (360°) should not sweep through all hues
        h_stops = hsv_stops[:, 0]
        for i in range(1, len(h_stops)):
            # If hue gap > 0.5 (180°), take shorter path by wrapping
            diff = h_stops[i] - h_stops[i-1]
            if diff > 0.5:
                h_stops[i] -= 1.0
            elif diff < -0.5:
                h_stops[i] += 1.0

        # Create sample points
        positions = np.linspace(0.0, 1.0, len(colors), dtype=np.float32)
        samples = np.linspace(0.0, 1.0, size, dtype=np.float32)

        # Interpolate each HSV channel independently
        h = np.interp(samples, positions, h_stops) % 1.0  # Wrap hue to [0, 1]
        s = np.interp(samples, positions, hsv_stops[:, 1])
        v = np.interp(samples, positions, hsv_stops[:, 2])

        # Convert back to RGB
        lut = np.array([colorsys.hsv_to_rgb(h[i], s[i], v[i]) for i in range(size)], dtype=np.float32)

        return lut

    def _boost_saturation(self, rgb: np.ndarray, factor: float) -> np.ndarray:
        """Boost saturation of RGB array.

        Args:
            rgb: RGB array (H, W, 3) or (N, 3) in [0, 1]
            factor: Saturation multiplier (1.0 = no change, >1.0 = more saturated)

        Returns:
            RGB array with adjusted saturation
        """
        if factor == 1.0:
            return rgb

        # Convert to HSV, boost saturation, convert back
        # Using simple RGB->HSV conversion
        shape = rgb.shape
        rgb_flat = rgb.reshape(-1, 3)

        # Compute grayscale (luminance)
        gray = 0.299 * rgb_flat[:, 0] + 0.587 * rgb_flat[:, 1] + 0.114 * rgb_flat[:, 2]
        gray = gray[:, np.newaxis]  # (N, 1)

        # Interpolate between grayscale and original color
        # factor = 1.0: original color
        # factor = 0.0: grayscale
        # factor > 1.0: oversaturated
        result = gray + factor * (rgb_flat - gray)
        result = np.clip(result, 0.0, 1.0)

        return result.reshape(shape)

    def _apply_palette(self) -> np.ndarray:
        """Apply current palette and adjustments to LIC array.

        Returns:
            RGB array (H, W, 3) uint8
        """
        # Use Lospec palette if active, otherwise use library palette
        if self.lospec_palette is not None:
            lut_numpy = self.lospec_palette_lut
        else:
            lut_numpy = _get_palette_lut(self.current_palette)

        # Apply saturation boost to LUT
        if self.saturation != 1.0:
            lut_numpy = self._boost_saturation(lut_numpy, self.saturation)

        if self.use_gpu:
            # GPU path (fast!)
            lut_tensor = GPUContext.to_gpu(lut_numpy)

            # Use proper clipping like main app
            from flowcol import defaults
            rgb_tensor = build_base_rgb_gpu(
                self.lic_tensor,
                clip_percent=defaults.DEFAULT_CLIP_PERCENT,
                brightness=self.brightness,
                contrast=self.contrast,
                gamma=self.gamma,
                color_enabled=True,
                lut=lut_tensor,
            )

            rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)
            rgb_array = GPUContext.to_cpu(rgb_tensor)
        else:
            # CPU path (slower fallback)
            lut = lut_numpy

            # Normalize and apply brightness/contrast/gamma (match GPU behavior)
            lic = self.lic_array.copy()

            # Percentile clipping (match GPU path)
            from flowcol import defaults
            clip_percent = defaults.DEFAULT_CLIP_PERCENT
            if clip_percent > 0:
                lower = clip_percent / 100.0
                upper = 1.0 - lower
                vmin, vmax = np.percentile(lic, [lower * 100, upper * 100])
                if vmax > vmin:
                    lic = np.clip((lic - vmin) / (vmax - vmin), 0.0, 1.0)
                else:
                    lic = np.clip(lic, 0.0, 1.0)

            # Contrast adjustment: (val - 0.5) * contrast + 0.5
            if self.contrast != 1.0:
                lic = (lic - 0.5) * self.contrast + 0.5
                lic = np.clip(lic, 0.0, 1.0)

            # Brightness adjustment (additive, not multiplicative!)
            if self.brightness != 0.0:
                lic = lic + self.brightness
                lic = np.clip(lic, 0.0, 1.0)

            # Gamma correction
            if self.gamma != 1.0:
                lic = np.power(lic, self.gamma)
                lic = np.clip(lic, 0.0, 1.0)

            # Apply LUT
            indices = (lic * (len(lut) - 1)).astype(np.int32)
            indices = np.clip(indices, 0, len(lut) - 1)
            rgb_array = lut[indices]

        # Convert to uint8
        rgb_uint8 = (rgb_array * 255).astype(np.uint8)
        return rgb_uint8

    def _update_texture(self) -> None:
        """Recompute RGB and update DPG texture."""
        rgb_uint8 = self._apply_palette()
        pil_img = Image.fromarray(rgb_uint8, mode='RGB')

        width, height, data = self._image_to_texture_data(pil_img)

        # Create or update texture
        if self.texture_id is None:
            self.texture_id = dpg.add_dynamic_texture(
                width, height, data, parent=self.texture_registry_id
            )
        else:
            dpg.set_value(self.texture_id, data)

    def _on_palette_change(self, sender, app_data, user_data) -> None:
        """Callback when palette selection changes."""
        self.current_palette = user_data
        self._update_texture()
        # Close popup
        if self.palette_popup_id is not None:
            dpg.configure_item(self.palette_popup_id, show=False)

    def _on_brightness_change(self, sender, app_data) -> None:
        """Callback when brightness slider changes."""
        self.brightness = app_data
        self._update_texture()

    def _on_contrast_change(self, sender, app_data) -> None:
        """Callback when contrast slider changes."""
        self.contrast = app_data
        self._update_texture()

    def _on_gamma_change(self, sender, app_data) -> None:
        """Callback when gamma slider changes."""
        self.gamma = app_data
        self._update_texture()

    def _on_saturation_change(self, sender, app_data) -> None:
        """Callback when saturation slider changes."""
        self.saturation = app_data
        self._update_texture()

    def _on_max_lospec_colors_change(self, sender, app_data) -> None:
        """Callback when max Lospec colors slider changes."""
        self.max_lospec_colors = int(app_data)

    def _on_randomize(self) -> None:
        """Randomize palette selection."""
        palette_names = list(_RUNTIME_PALETTES.keys())
        if not palette_names:
            return

        import random
        # Pick random palette different from current
        candidates = [p for p in palette_names if p != self.current_palette]
        if candidates:
            self.current_palette = random.choice(candidates)
        else:
            self.current_palette = palette_names[0]

        self._update_texture()
        dpg.set_value("current_palette_text", f"Palette: {self.current_palette}")

    def _on_try_random_lospec(self) -> None:
        """Fetch and apply a random Lospec palette."""
        try:
            print("Fetching random Lospec palette...")
            palette_data = fetch_random_palette(timeout=5.0)

            original_color_count = len(palette_data['colors'])
            colors_array = np.array(palette_data['colors'], dtype=np.float32)

            # Skip palettes that are too large (>16 colors)
            # These lose artistic intent when sampled
            if len(colors_array) > 16:
                print(f"  Skipping - too many colors ({original_color_count}), fetching another...")
                self._on_try_random_lospec()  # Retry
                return

            print(f"  Using all {len(colors_array)} colors (no sampling)")

            # Sort by luminance (darkest to brightest)
            luminances = np.mean(colors_array, axis=1)
            sort_indices = np.argsort(luminances)
            colors_array = colors_array[sort_indices]

            # Only extend with black/white if luminance range is insufficient
            lum_min, lum_max = luminances.min(), luminances.max()
            if lum_min > 0.1 or lum_max < 0.9:
                black = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                colors_array = np.vstack([black, colors_array, white])
                print(f"  Extended with black/white (lum range {lum_min:.2f}-{lum_max:.2f} → 0.0-1.0)")
            else:
                print(f"  No extension needed (lum range {lum_min:.2f}-{lum_max:.2f})")

            # Build LUT using HSV interpolation to preserve color vibrancy
            self.lospec_palette_lut = self._build_palette_lut_hsv(colors_array)
            self.lospec_palette = palette_data

            # Update display
            self._update_texture()

            # Update UI text
            dpg.set_value("current_palette_text",
                         f"Lospec: {palette_data['name']}\nby {palette_data['author']} ({len(colors_array)}/{original_color_count} colors)")

            print(f"✓ Loaded: {palette_data['name']} by {palette_data['author']} ({len(colors_array)}/{original_color_count} colors)")

        except Exception as e:
            print(f"✗ Failed to fetch Lospec palette: {e}")

    def _on_save_lospec_palette(self) -> None:
        """Save current Lospec palette to library."""
        if self.lospec_palette is None:
            print("ℹ No Lospec palette to save (use 'Try Random Lospec' first)")
            return

        palette_data = self.lospec_palette
        add_palette(palette_data['name'], palette_data['colors'])
        print(f"✓ Saved '{palette_data['name']}' to library")

        # Clear Lospec palette and switch to saved version
        self.lospec_palette = None
        self.lospec_palette_lut = None
        self.current_palette = palette_data['name']
        dpg.set_value("current_palette_text", f"Palette: {self.current_palette}")

    def _on_save_palette(self) -> None:
        """Save current palette to library (if it's not already there)."""
        if self.current_palette not in _RUNTIME_PALETTES:
            print(f"✗ Palette '{self.current_palette}' not found in runtime palettes")
            return

        # Already in library
        print(f"ℹ Palette '{self.current_palette}' is already in your library")

    def _build_palette_selector_popup(self) -> None:
        """Build palette selector popup with colormap previews."""
        if self.palette_popup_id is not None:
            return  # Already created

        with dpg.window(
            label="Select Palette",
            modal=True,
            show=False,
            tag="palette_popup",
            no_resize=True,
            width=420,
            height=600,
            no_open_over_existing_popup=False,
        ) as popup:
            self.palette_popup_id = popup

            dpg.add_text("Click a palette to apply:")
            dpg.add_separator()

            with dpg.child_window(height=520):
                for palette_name in sorted(_RUNTIME_PALETTES.keys()):
                    # Use colormap_button like main FlowCol app
                    if palette_name in self.palette_colormaps:
                        colormap_tag = self.palette_colormaps[palette_name]
                        btn = dpg.add_colormap_button(
                            label=palette_name,
                            width=380,
                            height=25,
                            callback=self._on_palette_change,
                            user_data=palette_name,
                        )
                        dpg.bind_colormap(btn, colormap_tag)
                    else:
                        # Fallback for palettes without colormaps
                        dpg.add_button(
                            label=palette_name,
                            width=380,
                            callback=self._on_palette_change,
                            user_data=palette_name,
                        )

                    dpg.add_spacer(height=2)

    def _open_palette_selector(self) -> None:
        """Open palette selector popup."""
        self._build_palette_selector_popup()
        # Center modal
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        modal_width = 420
        modal_height = 600
        pos_x = (viewport_width - modal_width) // 2
        pos_y = (viewport_height - modal_height) // 2
        dpg.configure_item(self.palette_popup_id, pos=[pos_x, pos_y], show=True)

    def _build_ui(self) -> None:
        """Build DearPyGui interface."""
        dpg.create_context()

        # Create texture registry
        self.texture_registry_id = dpg.add_texture_registry()

        # Create colormap registry for palette previews
        self.colormap_registry_id = dpg.add_colormap_registry()
        for palette_name, colors_normalized in _RUNTIME_PALETTES.items():
            colors_255 = [
                [int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255]
                for c in colors_normalized
            ]
            tag = f"colormap_{palette_name.replace(' ', '_').replace('&', 'and')}"
            dpg.add_colormap(
                colors_255,
                qualitative=False,
                tag=tag,
                parent=self.colormap_registry_id
            )
            self.palette_colormaps[palette_name] = tag

        # Initial texture
        self._update_texture()

        # Calculate display size (fit image to reasonable screen space)
        max_display_size = 900  # Max pixels for display
        scale = min(1.0, max_display_size / max(self.width, self.height))
        display_width = int(self.width * scale)
        display_height = int(self.height * scale)

        # Main window
        with dpg.window(label="Palette Explorer", tag="main_window"):
            with dpg.group(horizontal=True):
                # Left panel: Render display
                with dpg.child_window(width=display_width + 40, height=display_height + 60):
                    dpg.add_text("Rendered Output:")
                    dpg.add_image(self.texture_id, width=display_width, height=display_height)

                # Right panel: Controls
                with dpg.child_window(width=380):
                    dpg.add_text("Palette Explorer", color=(100, 200, 255))
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    # Palette selector
                    dpg.add_text(f"Palette: {self.current_palette}", tag="current_palette_text")
                    dpg.add_button(
                        label="Change Palette",
                        width=200,
                        callback=lambda: self._open_palette_selector(),
                    )
                    dpg.add_button(
                        label="🎲 Randomize Library",
                        width=200,
                        callback=lambda: self._on_randomize(),
                    )
                    dpg.add_spacer(height=5)
                    dpg.add_separator()
                    dpg.add_spacer(height=5)
                    dpg.add_text("Lospec Integration:")
                    dpg.add_slider_int(
                        label="Max Colors",
                        default_value=self.max_lospec_colors,
                        min_value=3,
                        max_value=32,
                        callback=self._on_max_lospec_colors_change,
                        width=200,
                    )
                    dpg.add_button(
                        label="🌐 Try Random Lospec",
                        width=200,
                        callback=lambda: self._on_try_random_lospec(),
                    )
                    dpg.add_button(
                        label="💾 Save Lospec to Library",
                        width=200,
                        callback=lambda: self._on_save_lospec_palette(),
                    )
                    dpg.add_spacer(height=20)
                    dpg.add_separator()
                    dpg.add_spacer(height=20)

                    # Adjustment sliders
                    dpg.add_text("Adjustments:")
                    dpg.add_slider_float(
                        label="Brightness",
                        default_value=self.brightness,
                        min_value=-0.5,
                        max_value=0.5,
                        callback=self._on_brightness_change,
                        width=250,
                        format="%.2f",
                    )
                    dpg.add_slider_float(
                        label="Contrast",
                        default_value=self.contrast,
                        min_value=0.5,
                        max_value=2.0,
                        callback=self._on_contrast_change,
                        width=250,
                        format="%.2f",
                    )
                    dpg.add_slider_float(
                        label="Gamma",
                        default_value=self.gamma,
                        min_value=0.5,
                        max_value=2.5,
                        callback=self._on_gamma_change,
                        width=250,
                        format="%.2f",
                    )
                    dpg.add_slider_float(
                        label="Saturation",
                        default_value=self.saturation,
                        min_value=0.0,
                        max_value=2.0,
                        callback=self._on_saturation_change,
                        width=250,
                        format="%.2f",
                    )

                    dpg.add_spacer(height=20)
                    dpg.add_separator()
                    dpg.add_spacer(height=20)

                    # Info
                    dpg.add_text("Info:")
                    dpg.add_text(f"Resolution: {self.width} x {self.height}")
                    dpg.add_text(f"GPU: {'Yes' if self.use_gpu else 'No'}")
                    dpg.add_text(f"Palettes: {len(_RUNTIME_PALETTES)}")

        # Set viewport size to fit content
        viewport_width = display_width + 380 + 100  # image + controls + padding
        viewport_height = max(display_height + 100, 750)  # ensure minimum height for controls

        dpg.create_viewport(
            title="FlowCol Palette Explorer",
            width=viewport_width,
            height=viewport_height,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

    def run(self) -> None:
        """Launch the palette explorer."""
        self._build_ui()
        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    """Entry point."""
    if len(sys.argv) != 2:
        print("Usage: python tools/palette_explorer.py path/to/project.flowcol")
        sys.exit(1)

    project_path = sys.argv[1]

    try:
        explorer = PaletteExplorer(project_path)
        explorer.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
