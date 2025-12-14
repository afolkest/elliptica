"""Standalone OKLCH color picker prototype.

A minimal picker to explore OKLCH color space with:
- L/C/H sliders
- 2D slice visualization (C × H at fixed L)
- Gamut boundary visualization
- sRGB preview

Usage:
    python tools/oklch_picker.py
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for elliptica imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from elliptica.colorspace.oklch import oklch_to_srgb
from elliptica.colorspace.gamut import max_chroma_fast

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    print("Error: dearpygui not installed. Install with: pip install dearpygui")
    sys.exit(1)


class OklchPicker:
    """OKLCH color picker with 2D slice visualization."""

    def __init__(self):
        # Current color state
        self.L = 0.70
        self.C = 0.10
        self.H = 270.0

        # 2D slice dimensions
        self.slice_width = 360  # 1 pixel per degree of hue
        self.slice_height = 128  # Chroma resolution
        self.c_max_display = 0.4  # Max chroma shown in slice

        # Texture IDs
        self.texture_registry_id = None
        self.slice_texture_id = None
        self.preview_texture_id = None
        self.l_gradient_texture_id = None
        self.h_gradient_texture_id = None

        # UI element IDs
        self.slice_drawlist_id = None

        # Precompute slice arrays (reused on updates)
        self._h_grid = np.linspace(0, 360, self.slice_width, endpoint=False, dtype=np.float32)
        self._c_grid = np.linspace(0, self.c_max_display, self.slice_height, dtype=np.float32)
        self._H_mesh, self._C_mesh = np.meshgrid(self._h_grid, self._c_grid)

    def _generate_ch_slice(self, L: float) -> np.ndarray:
        """Generate C×H slice texture at fixed L.

        Returns RGBA array (H, W, 4) with gamut-aware coloring.
        Out-of-gamut regions rendered as dark gray.
        """
        # Broadcast L to match mesh shape
        L_arr = np.full_like(self._C_mesh, L)

        # Get max chroma for each (L, H) pair
        max_c = max_chroma_fast(L_arr, self._H_mesh)

        # Compute RGB for all pixels
        rgb = oklch_to_srgb(L_arr, self._C_mesh, self._H_mesh)

        # Create RGBA output
        rgba = np.ones((self.slice_height, self.slice_width, 4), dtype=np.float32)

        # Mask for in-gamut pixels
        in_gamut = self._C_mesh <= max_c

        # Set RGB values (clamp for safety)
        rgba[..., :3] = np.clip(rgb, 0.0, 1.0)

        # Out-of-gamut: dark gray with reduced alpha
        out_of_gamut = ~in_gamut
        rgba[out_of_gamut, 0] = 0.15
        rgba[out_of_gamut, 1] = 0.15
        rgba[out_of_gamut, 2] = 0.15
        rgba[out_of_gamut, 3] = 0.5

        return rgba

    def _generate_l_gradient(self) -> np.ndarray:
        """Generate L gradient texture at current C, H."""
        width = 360
        height = 16

        L_arr = np.linspace(0, 1, width, dtype=np.float32)
        C_arr = np.full(width, self.C, dtype=np.float32)
        H_arr = np.full(width, self.H, dtype=np.float32)

        # Check gamut
        max_c = max_chroma_fast(L_arr, H_arr)
        in_gamut = C_arr <= max_c

        rgb = oklch_to_srgb(L_arr, C_arr, H_arr)
        rgb = np.clip(rgb, 0.0, 1.0)

        # Out of gamut: show gray
        rgb[~in_gamut] = [0.2, 0.2, 0.2]

        # Expand to 2D
        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = rgb[np.newaxis, :, :]

        return rgba

    def _generate_h_gradient(self) -> np.ndarray:
        """Generate H gradient texture at current L, C."""
        width = 360
        height = 16

        H_arr = np.linspace(0, 360, width, endpoint=False, dtype=np.float32)
        L_arr = np.full(width, self.L, dtype=np.float32)
        C_arr = np.full(width, self.C, dtype=np.float32)

        # Check gamut
        max_c = max_chroma_fast(L_arr, H_arr)
        in_gamut = C_arr <= max_c

        rgb = oklch_to_srgb(L_arr, C_arr, H_arr)
        rgb = np.clip(rgb, 0.0, 1.0)

        # Out of gamut: show gray
        rgb[~in_gamut] = [0.2, 0.2, 0.2]

        # Expand to 2D
        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = rgb[np.newaxis, :, :]

        return rgba

    def _generate_preview(self) -> np.ndarray:
        """Generate preview swatch texture."""
        size = 80
        rgb = oklch_to_srgb(
            np.array(self.L, dtype=np.float32),
            np.array(self.C, dtype=np.float32),
            np.array(self.H, dtype=np.float32),
        )
        rgb = np.clip(rgb, 0.0, 1.0)

        rgba = np.ones((size, size, 4), dtype=np.float32)
        rgba[..., :3] = rgb

        return rgba

    def _get_current_rgb(self) -> tuple[int, int, int]:
        """Get current color as RGB 0-255."""
        rgb = oklch_to_srgb(
            np.array(self.L, dtype=np.float32),
            np.array(self.C, dtype=np.float32),
            np.array(self.H, dtype=np.float32),
        )
        rgb = np.clip(rgb, 0.0, 1.0)
        return tuple(int(c * 255) for c in rgb.flat)

    def _get_max_chroma(self) -> float:
        """Get max valid chroma for current L, H."""
        return float(max_chroma_fast(
            np.array(self.L, dtype=np.float32),
            np.array(self.H, dtype=np.float32),
        ))

    def _is_in_gamut(self) -> bool:
        """Check if current color is in sRGB gamut."""
        return self.C <= self._get_max_chroma()

    def _update_slice_texture(self):
        """Regenerate and upload C×H slice texture."""
        rgba = self._generate_ch_slice(self.L)
        data = rgba.ravel()

        if self.slice_texture_id is None:
            self.slice_texture_id = dpg.add_dynamic_texture(
                self.slice_width, self.slice_height, data,
                parent=self.texture_registry_id
            )
        else:
            dpg.set_value(self.slice_texture_id, data)

    def _update_gradient_textures(self):
        """Regenerate L and H gradient textures."""
        # L gradient
        l_rgba = self._generate_l_gradient()
        l_data = l_rgba.ravel()
        if self.l_gradient_texture_id is None:
            self.l_gradient_texture_id = dpg.add_dynamic_texture(
                360, 16, l_data, parent=self.texture_registry_id
            )
        else:
            dpg.set_value(self.l_gradient_texture_id, l_data)

        # H gradient
        h_rgba = self._generate_h_gradient()
        h_data = h_rgba.ravel()
        if self.h_gradient_texture_id is None:
            self.h_gradient_texture_id = dpg.add_dynamic_texture(
                360, 16, h_data, parent=self.texture_registry_id
            )
        else:
            dpg.set_value(self.h_gradient_texture_id, h_data)

    def _update_preview_texture(self):
        """Regenerate preview swatch texture."""
        rgba = self._generate_preview()
        data = rgba.ravel()

        if self.preview_texture_id is None:
            self.preview_texture_id = dpg.add_dynamic_texture(
                80, 80, data, parent=self.texture_registry_id
            )
        else:
            dpg.set_value(self.preview_texture_id, data)

    def _update_slice_crosshair(self):
        """Draw crosshair on slice at current C, H position."""
        if self.slice_drawlist_id is None:
            return

        dpg.delete_item(self.slice_drawlist_id, children_only=True)

        # Draw the slice image
        dpg.draw_image(
            self.slice_texture_id,
            (0, 0),
            (self.slice_width, self.slice_height),
            parent=self.slice_drawlist_id,
        )

        # Crosshair position
        x = self.H  # H maps directly to x (0-360)
        y = (self.C / self.c_max_display) * self.slice_height  # C maps to y

        # Draw crosshair
        color = (255, 255, 255, 200)
        thickness = 1.5

        # Horizontal line
        dpg.draw_line((0, y), (self.slice_width, y), color=color, thickness=thickness,
                      parent=self.slice_drawlist_id)
        # Vertical line
        dpg.draw_line((x, 0), (x, self.slice_height), color=color, thickness=thickness,
                      parent=self.slice_drawlist_id)

        # Center circle
        dpg.draw_circle((x, y), 5, color=color, thickness=2, parent=self.slice_drawlist_id)

        # Draw gamut boundary line
        gamut_points = []
        for h_idx, h in enumerate(self._h_grid):
            max_c = float(max_chroma_fast(
                np.array(self.L, dtype=np.float32),
                np.array(h, dtype=np.float32),
            ))
            y_boundary = (max_c / self.c_max_display) * self.slice_height
            gamut_points.append((h_idx, min(y_boundary, self.slice_height)))

        if len(gamut_points) > 1:
            dpg.draw_polyline(gamut_points, color=(255, 255, 255, 150), thickness=1.5,
                             parent=self.slice_drawlist_id)

    def _update_info_text(self):
        """Update info display."""
        r, g, b = self._get_current_rgb()
        max_c = self._get_max_chroma()
        in_gamut = self._is_in_gamut()

        gamut_status = "✓ In gamut" if in_gamut else "⚠ Out of gamut"
        gamut_color = (100, 255, 100) if in_gamut else (255, 150, 100)

        dpg.set_value("rgb_text", f"RGB: {r}, {g}, {b}")
        dpg.set_value("hex_text", f"Hex: #{r:02x}{g:02x}{b:02x}")
        dpg.set_value("max_c_text", f"Max C at this L,H: {max_c:.3f}")
        dpg.set_value("gamut_text", gamut_status)
        dpg.configure_item("gamut_text", color=gamut_color)

    def _refresh_all(self):
        """Full refresh of all visuals."""
        self._update_slice_texture()
        self._update_gradient_textures()
        self._update_preview_texture()
        self._update_slice_crosshair()
        self._update_info_text()

    def _on_l_change(self, sender, value):
        """Handle L slider change."""
        self.L = value
        self._refresh_all()

    def _on_c_change(self, sender, value):
        """Handle C slider change."""
        self.C = value
        self._update_gradient_textures()
        self._update_preview_texture()
        self._update_slice_crosshair()
        self._update_info_text()

    def _on_h_change(self, sender, value):
        """Handle H slider change."""
        self.H = value
        self._update_gradient_textures()
        self._update_preview_texture()
        self._update_slice_crosshair()
        self._update_info_text()

    def _on_slice_click(self, sender, app_data):
        """Handle click in the slice area."""
        mouse_pos = dpg.get_mouse_pos(local=False)

        # Get drawlist position
        if not dpg.does_item_exist(self.slice_drawlist_id):
            return

        rect_min = dpg.get_item_rect_min(self.slice_drawlist_id)
        if rect_min is None:
            return

        # Local coordinates
        local_x = mouse_pos[0] - rect_min[0]
        local_y = mouse_pos[1] - rect_min[1]

        # Bounds check
        if not (0 <= local_x < self.slice_width and 0 <= local_y < self.slice_height):
            return

        # Convert to H, C
        new_h = local_x  # Direct mapping
        new_c = (local_y / self.slice_height) * self.c_max_display

        self.H = float(np.clip(new_h, 0, 359.9))
        self.C = float(np.clip(new_c, 0, self.c_max_display))

        # Update sliders
        dpg.set_value("h_slider", self.H)
        dpg.set_value("c_slider", self.C)

        # Refresh visuals
        self._update_gradient_textures()
        self._update_preview_texture()
        self._update_slice_crosshair()
        self._update_info_text()

    def _clamp_to_gamut(self):
        """Clamp C to max valid value."""
        max_c = self._get_max_chroma()
        if self.C > max_c:
            self.C = max_c
            dpg.set_value("c_slider", self.C)
            self._update_gradient_textures()
            self._update_preview_texture()
            self._update_slice_crosshair()
            self._update_info_text()

    def _build_ui(self):
        """Build the DearPyGui interface."""
        dpg.create_context()

        # Texture registry
        self.texture_registry_id = dpg.add_texture_registry()

        # Initialize textures
        self._update_slice_texture()
        self._update_gradient_textures()
        self._update_preview_texture()

        # Mouse handler for slice clicks
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=self._on_slice_click)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,
                                       callback=self._on_slice_click, threshold=0.0)

        # Main window
        with dpg.window(label="OKLCH Color Picker", tag="main_window"):
            dpg.add_text("OKLCH Color Picker", color=(150, 200, 255))
            dpg.add_text("Click/drag in the C×H slice to pick colors", color=(150, 150, 150))
            dpg.add_separator()
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                # Left: Slice and sliders
                with dpg.group():
                    # C×H Slice at top
                    dpg.add_text("Chroma × Hue (at current L)")
                    self.slice_drawlist_id = dpg.add_drawlist(
                        width=self.slice_width,
                        height=self.slice_height,
                    )
                    self._update_slice_crosshair()

                    dpg.add_spacer(height=15)

                    # All three sliders together
                    # L slider with gradient
                    dpg.add_text("Lightness (L)")
                    dpg.add_image(self.l_gradient_texture_id, width=360, height=16)
                    dpg.add_slider_float(
                        label="",
                        default_value=self.L,
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                        width=360,
                        callback=self._on_l_change,
                        tag="l_slider",
                    )

                    dpg.add_spacer(height=8)

                    # C slider
                    dpg.add_text("Chroma (C)")
                    dpg.add_slider_float(
                        label="",
                        default_value=self.C,
                        min_value=0.0,
                        max_value=self.c_max_display,
                        format="%.4f",
                        width=360,
                        callback=self._on_c_change,
                        tag="c_slider",
                    )

                    dpg.add_spacer(height=8)

                    # H slider with gradient
                    dpg.add_text("Hue (H)")
                    dpg.add_image(self.h_gradient_texture_id, width=360, height=16)
                    dpg.add_slider_float(
                        label="",
                        default_value=self.H,
                        min_value=0.0,
                        max_value=360.0,
                        format="%.1f°",
                        width=360,
                        callback=self._on_h_change,
                        tag="h_slider",
                    )

                dpg.add_spacer(width=30)

                # Right: Preview and info
                with dpg.group():
                    dpg.add_text("Preview")
                    dpg.add_image(self.preview_texture_id, width=80, height=80)

                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    dpg.add_text("Color Values", color=(150, 200, 255))
                    dpg.add_text(f"L: {self.L:.3f}", tag="l_value_text")
                    dpg.add_text(f"C: {self.C:.4f}", tag="c_value_text")
                    dpg.add_text(f"H: {self.H:.1f}°", tag="h_value_text")

                    dpg.add_spacer(height=10)

                    r, g, b = self._get_current_rgb()
                    dpg.add_text(f"RGB: {r}, {g}, {b}", tag="rgb_text")
                    dpg.add_text(f"Hex: #{r:02x}{g:02x}{b:02x}", tag="hex_text")

                    dpg.add_spacer(height=10)

                    max_c = self._get_max_chroma()
                    dpg.add_text(f"Max C at this L,H: {max_c:.3f}", tag="max_c_text")
                    dpg.add_text("✓ In gamut", tag="gamut_text", color=(100, 255, 100))

                    dpg.add_spacer(height=20)

                    dpg.add_button(
                        label="Clamp to Gamut",
                        width=120,
                        callback=lambda: self._clamp_to_gamut(),
                    )

                    dpg.add_spacer(height=20)
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    dpg.add_text("Tips:", color=(150, 150, 150))
                    dpg.add_text("• White line = gamut boundary", color=(120, 120, 120))
                    dpg.add_text("• Gray area = out of sRGB", color=(120, 120, 120))
                    dpg.add_text("• Drag L slider to see", color=(120, 120, 120))
                    dpg.add_text("  gamut shape change", color=(120, 120, 120))

        # Viewport
        dpg.create_viewport(
            title="OKLCH Color Picker",
            width=620,
            height=520,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

    def run(self):
        """Launch the picker."""
        self._build_ui()
        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    picker = OklchPicker()
    picker.run()


if __name__ == "__main__":
    main()
