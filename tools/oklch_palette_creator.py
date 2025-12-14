"""OKLCH-based palette creator.

Create color palettes using the perceptually uniform OKLCH color space.
Stops are defined in OKLCH and interpolated in OKLCH for smooth gradients.

Usage:
    python tools/oklch_palette_creator.py
"""

import sys
from pathlib import Path
from typing import Optional
import json

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from elliptica.colorspace.oklch import oklch_to_srgb
from elliptica.colorspace.gamut import max_chroma_fast, gamut_map_to_srgb
from elliptica.render import add_palette, USER_PALETTES_PATH

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    print("Error: dearpygui not installed. Install with: pip install dearpygui")
    sys.exit(1)


class OklchPaletteCreator:
    """OKLCH-based palette creator with gradient stops."""

    def __init__(self):
        # Gradient stops: each is {id, pos, L, C, H}
        self.stops: list[dict] = []
        self._next_stop_id: int = 0
        self.selected_stop_id: Optional[int] = None
        self.is_dragging: bool = False

        # Layout
        self.gradient_width = 600
        self.gradient_height = 60
        self.gradient_bar_top = 10
        self.gradient_bar_bottom = 50
        self.gradient_bar_padding = 20
        self.handle_radius = 8

        # OKLCH picker dimensions
        self.slice_width = 360
        self.slice_height = 120
        self.c_max_display = 0.4

        # Precompute slice grids
        self._h_grid = np.linspace(0, 360, self.slice_width, endpoint=False, dtype=np.float32)
        self._c_grid = np.linspace(0, self.c_max_display, self.slice_height, dtype=np.float32)
        self._H_mesh, self._C_mesh = np.meshgrid(self._h_grid, self._c_grid)

        # Texture IDs
        self.texture_registry = None
        self.gradient_texture_id = None
        self.slice_texture_id = None
        self.preview_texture_id = None
        self.l_gradient_texture_id = None

        # UI element IDs
        self.gradient_drawlist_id = None
        self.slice_drawlist_id = None

        # Persistence
        self.palettes_path = USER_PALETTES_PATH.with_name("oklch_palettes.json")
        self.saved_palettes = self._load_palettes()

        # Initialize with default gradient
        self._init_default_gradient()

    def _load_palettes(self) -> dict[str, list[dict]]:
        """Load saved OKLCH palettes from disk."""
        if not self.palettes_path.exists():
            return {}
        try:
            with open(self.palettes_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: failed to load palettes: {e}")
            return {}

    def _save_palettes(self):
        """Save OKLCH palettes to disk."""
        self.palettes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.palettes_path, "w") as f:
            json.dump(self.saved_palettes, f, indent=2)

    def _init_default_gradient(self):
        """Initialize with a simple two-stop gradient."""
        self.stops = [
            {"id": 0, "pos": 0.0, "L": 0.20, "C": 0.0, "H": 0.0},
            {"id": 1, "pos": 1.0, "L": 0.95, "C": 0.0, "H": 0.0},
        ]
        self._next_stop_id = 2
        self.selected_stop_id = 0

    def _sort_stops(self):
        """Keep stops sorted by position."""
        self.stops.sort(key=lambda s: s["pos"])

    def _get_stop_index(self, stop_id: int) -> Optional[int]:
        """Get index of stop by ID."""
        for i, stop in enumerate(self.stops):
            if stop["id"] == stop_id:
                return i
        return None

    def _get_selected_stop(self) -> Optional[dict]:
        """Get currently selected stop."""
        if self.selected_stop_id is None:
            return None
        idx = self._get_stop_index(self.selected_stop_id)
        return self.stops[idx] if idx is not None else None

    # ---------------------------------------------------------------
    # OKLCH utilities
    # ---------------------------------------------------------------

    def _oklch_to_rgb_clamped(self, L: float, C: float, H: float) -> tuple[float, float, float]:
        """Convert OKLCH to RGB, clamping to gamut."""
        rgb = gamut_map_to_srgb(
            np.array(L, dtype=np.float32),
            np.array(C, dtype=np.float32),
            np.array(H, dtype=np.float32),
            method='compress',
        )
        rgb = np.clip(rgb, 0.0, 1.0)
        return tuple(float(c) for c in rgb.flat)

    def _oklch_to_rgb255(self, L: float, C: float, H: float) -> tuple[int, int, int]:
        """Convert OKLCH to RGB 0-255."""
        r, g, b = self._oklch_to_rgb_clamped(L, C, H)
        return (int(r * 255), int(g * 255), int(b * 255))

    def _get_max_chroma(self, L: float, H: float) -> float:
        """Get max valid chroma for L, H."""
        return float(max_chroma_fast(
            np.array(L, dtype=np.float32),
            np.array(H, dtype=np.float32),
        ))

    def _is_in_gamut(self, L: float, C: float, H: float) -> bool:
        """Check if OKLCH color is in sRGB gamut."""
        return C <= self._get_max_chroma(L, H)

    # ---------------------------------------------------------------
    # Gradient interpolation
    # ---------------------------------------------------------------

    def _interpolate_oklch(self, t: float) -> tuple[float, float, float]:
        """Interpolate gradient at position t (0-1).

        Uses linear interpolation in OKLCH space with hue wraparound.
        """
        if not self.stops:
            return (0.5, 0.0, 0.0)

        if len(self.stops) == 1:
            s = self.stops[0]
            return (s["L"], s["C"], s["H"])

        t = np.clip(t, 0.0, 1.0)

        # Find surrounding stops
        for i in range(len(self.stops) - 1):
            s0 = self.stops[i]
            s1 = self.stops[i + 1]
            if s0["pos"] <= t <= s1["pos"]:
                # Interpolate between s0 and s1
                if s1["pos"] == s0["pos"]:
                    frac = 0.0
                else:
                    frac = (t - s0["pos"]) / (s1["pos"] - s0["pos"])

                L = s0["L"] + frac * (s1["L"] - s0["L"])
                C = s0["C"] + frac * (s1["C"] - s0["C"])

                # Hue interpolation with wraparound
                h0, h1 = s0["H"], s1["H"]
                dh = h1 - h0
                if dh > 180:
                    dh -= 360
                elif dh < -180:
                    dh += 360
                H = (h0 + frac * dh) % 360

                return (L, C, H)

        # Fallback to last stop
        s = self.stops[-1]
        return (s["L"], s["C"], s["H"])

    def _build_lut(self, size: int = 256) -> np.ndarray:
        """Build RGB LUT from current gradient."""
        positions = np.linspace(0.0, 1.0, size, dtype=np.float32)

        # Interpolate in OKLCH
        L_arr = np.empty(size, dtype=np.float32)
        C_arr = np.empty(size, dtype=np.float32)
        H_arr = np.empty(size, dtype=np.float32)

        for i, t in enumerate(positions):
            L, C, H = self._interpolate_oklch(float(t))
            L_arr[i] = L
            C_arr[i] = C
            H_arr[i] = H

        # Convert to RGB
        rgb = gamut_map_to_srgb(L_arr, C_arr, H_arr, method='compress')
        return np.clip(rgb, 0.0, 1.0)

    # ---------------------------------------------------------------
    # Texture generation
    # ---------------------------------------------------------------

    def _generate_gradient_texture(self) -> np.ndarray:
        """Generate gradient bar texture."""
        lut = self._build_lut(size=self.gradient_width)
        height = 32
        rgba = np.ones((height, self.gradient_width, 4), dtype=np.float32)
        rgba[:, :, :3] = lut[np.newaxis, :, :]
        return rgba

    def _generate_ch_slice(self, L: float) -> np.ndarray:
        """Generate C x H slice at fixed L."""
        L_arr = np.full_like(self._C_mesh, L)
        max_c = max_chroma_fast(L_arr, self._H_mesh)

        rgb = gamut_map_to_srgb(L_arr, self._C_mesh, self._H_mesh, method='compress')

        rgba = np.ones((self.slice_height, self.slice_width, 4), dtype=np.float32)
        rgba[..., :3] = np.clip(rgb, 0.0, 1.0)

        # Gray out of gamut
        out_of_gamut = self._C_mesh > max_c
        rgba[out_of_gamut, 0] = 0.15
        rgba[out_of_gamut, 1] = 0.15
        rgba[out_of_gamut, 2] = 0.15
        rgba[out_of_gamut, 3] = 0.5

        return rgba

    def _generate_l_gradient(self, C: float, H: float) -> np.ndarray:
        """Generate L gradient bar."""
        width = 360
        height = 16

        L_arr = np.linspace(0, 1, width, dtype=np.float32)
        C_arr = np.full(width, C, dtype=np.float32)
        H_arr = np.full(width, H, dtype=np.float32)

        max_c = max_chroma_fast(L_arr, H_arr)
        in_gamut = C_arr <= max_c

        rgb = gamut_map_to_srgb(L_arr, C_arr, H_arr, method='compress')
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb[~in_gamut] = [0.2, 0.2, 0.2]

        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = rgb[np.newaxis, :, :]
        return rgba

    def _generate_preview(self, L: float, C: float, H: float) -> np.ndarray:
        """Generate preview swatch."""
        size = 60
        r, g, b = self._oklch_to_rgb_clamped(L, C, H)
        rgba = np.ones((size, size, 4), dtype=np.float32)
        rgba[..., :3] = [r, g, b]
        return rgba

    # ---------------------------------------------------------------
    # Texture updates
    # ---------------------------------------------------------------

    def _update_gradient_texture(self):
        """Update gradient bar texture."""
        rgba = self._generate_gradient_texture()
        data = rgba.ravel()
        if self.gradient_texture_id is None:
            self.gradient_texture_id = dpg.add_dynamic_texture(
                self.gradient_width, 32, data, parent=self.texture_registry
            )
        else:
            dpg.set_value(self.gradient_texture_id, data)

    def _update_slice_texture(self):
        """Update C x H slice texture for selected stop's L."""
        stop = self._get_selected_stop()
        L = stop["L"] if stop else 0.5

        rgba = self._generate_ch_slice(L)
        data = rgba.ravel()
        if self.slice_texture_id is None:
            self.slice_texture_id = dpg.add_dynamic_texture(
                self.slice_width, self.slice_height, data, parent=self.texture_registry
            )
        else:
            dpg.set_value(self.slice_texture_id, data)

    def _update_l_gradient_texture(self):
        """Update L gradient texture."""
        stop = self._get_selected_stop()
        C = stop["C"] if stop else 0.0
        H = stop["H"] if stop else 0.0

        rgba = self._generate_l_gradient(C, H)
        data = rgba.ravel()
        if self.l_gradient_texture_id is None:
            self.l_gradient_texture_id = dpg.add_dynamic_texture(
                360, 16, data, parent=self.texture_registry
            )
        else:
            dpg.set_value(self.l_gradient_texture_id, data)

    def _update_preview_texture(self):
        """Update preview swatch."""
        stop = self._get_selected_stop()
        L = stop["L"] if stop else 0.5
        C = stop["C"] if stop else 0.0
        H = stop["H"] if stop else 0.0

        rgba = self._generate_preview(L, C, H)
        data = rgba.ravel()
        if self.preview_texture_id is None:
            self.preview_texture_id = dpg.add_dynamic_texture(
                60, 60, data, parent=self.texture_registry
            )
        else:
            dpg.set_value(self.preview_texture_id, data)

    # ---------------------------------------------------------------
    # Gradient bar drawing
    # ---------------------------------------------------------------

    def _stop_to_x(self, pos: float) -> float:
        """Convert stop position to x coordinate."""
        bar_left = self.gradient_bar_padding
        bar_right = self.gradient_width - self.gradient_bar_padding
        return bar_left + pos * (bar_right - bar_left)

    def _x_to_pos(self, x: float) -> float:
        """Convert x coordinate to stop position."""
        bar_left = self.gradient_bar_padding
        bar_right = self.gradient_width - self.gradient_bar_padding
        return float(np.clip((x - bar_left) / (bar_right - bar_left), 0.0, 1.0))

    def _update_gradient_drawlist(self):
        """Redraw gradient bar and handles."""
        if self.gradient_drawlist_id is None:
            return

        dpg.delete_item(self.gradient_drawlist_id, children_only=True)

        bar_left = self.gradient_bar_padding
        bar_right = self.gradient_width - self.gradient_bar_padding

        # Draw gradient image
        dpg.draw_image(
            self.gradient_texture_id,
            (bar_left, self.gradient_bar_top),
            (bar_right, self.gradient_bar_bottom),
            parent=self.gradient_drawlist_id,
        )

        # Border
        dpg.draw_rectangle(
            (bar_left, self.gradient_bar_top),
            (bar_right, self.gradient_bar_bottom),
            color=(200, 200, 200, 255),
            thickness=1,
            parent=self.gradient_drawlist_id,
        )

        # Draw handles
        handle_base = self.gradient_bar_bottom + 4
        handle_height = 14

        for stop in self.stops:
            x = self._stop_to_x(stop["pos"])
            r, g, b = self._oklch_to_rgb255(stop["L"], stop["C"], stop["H"])
            fill = (r, g, b, 255)
            outline = (255, 230, 120, 255) if stop["id"] == self.selected_stop_id else (30, 30, 30, 255)

            # Triangle handle pointing up
            points = [
                (x, handle_base),
                (x - self.handle_radius, handle_base + handle_height),
                (x + self.handle_radius, handle_base + handle_height),
            ]
            dpg.draw_triangle(
                points[0], points[1], points[2],
                color=outline,
                fill=fill,
                thickness=2,
                parent=self.gradient_drawlist_id,
            )

    def _update_slice_crosshair(self):
        """Draw crosshair on C x H slice."""
        if self.slice_drawlist_id is None:
            return

        dpg.delete_item(self.slice_drawlist_id, children_only=True)

        # Draw slice image
        dpg.draw_image(
            self.slice_texture_id,
            (0, 0),
            (self.slice_width, self.slice_height),
            parent=self.slice_drawlist_id,
        )

        stop = self._get_selected_stop()
        if stop is None:
            return

        # Crosshair position
        x = stop["H"]  # H maps directly to x (0-360)
        y = (stop["C"] / self.c_max_display) * self.slice_height

        color = (255, 255, 255, 200)

        # Lines
        dpg.draw_line((0, y), (self.slice_width, y), color=color, thickness=1.5,
                     parent=self.slice_drawlist_id)
        dpg.draw_line((x, 0), (x, self.slice_height), color=color, thickness=1.5,
                     parent=self.slice_drawlist_id)

        # Center circle
        dpg.draw_circle((x, y), 5, color=color, thickness=2, parent=self.slice_drawlist_id)

        # Draw gamut boundary
        stop_L = stop["L"]
        gamut_points = []
        for h_idx, h in enumerate(self._h_grid):
            max_c = float(max_chroma_fast(
                np.array(stop_L, dtype=np.float32),
                np.array(h, dtype=np.float32),
            ))
            y_boundary = (max_c / self.c_max_display) * self.slice_height
            gamut_points.append((h_idx, min(y_boundary, self.slice_height)))

        if len(gamut_points) > 1:
            dpg.draw_polyline(gamut_points, color=(255, 255, 255, 150), thickness=1.5,
                             parent=self.slice_drawlist_id)

    # ---------------------------------------------------------------
    # UI sync
    # ---------------------------------------------------------------

    def _sync_ui_from_stop(self):
        """Sync UI widgets to selected stop."""
        stop = self._get_selected_stop()
        if stop is None:
            dpg.configure_item("l_slider", enabled=False)
            dpg.configure_item("c_slider", enabled=False)
            dpg.configure_item("h_slider", enabled=False)
            dpg.set_value("stop_info_text", "No stop selected")
            return

        dpg.configure_item("l_slider", enabled=True)
        dpg.configure_item("c_slider", enabled=True)
        dpg.configure_item("h_slider", enabled=True)

        dpg.set_value("l_slider", stop["L"])
        dpg.set_value("c_slider", stop["C"])
        dpg.set_value("h_slider", stop["H"])

        idx = self._get_stop_index(stop["id"])
        r, g, b = self._oklch_to_rgb255(stop["L"], stop["C"], stop["H"])
        in_gamut = self._is_in_gamut(stop["L"], stop["C"], stop["H"])
        gamut_str = "in gamut" if in_gamut else "OUT OF GAMUT"

        dpg.set_value("stop_info_text",
            f"Stop {idx + 1}/{len(self.stops)} @ {stop['pos']:.2f}\n"
            f"L={stop['L']:.3f}  C={stop['C']:.3f}  H={stop['H']:.1f}\n"
            f"RGB: {r}, {g}, {b}  ({gamut_str})"
        )

    def _refresh_all(self):
        """Full refresh of all visuals."""
        self._update_gradient_texture()
        self._update_slice_texture()
        self._update_l_gradient_texture()
        self._update_preview_texture()
        self._update_gradient_drawlist()
        self._update_slice_crosshair()
        self._sync_ui_from_stop()

    def _refresh_picker(self):
        """Refresh just the picker visuals (not gradient bar)."""
        self._update_slice_texture()
        self._update_l_gradient_texture()
        self._update_preview_texture()
        self._update_slice_crosshair()
        self._sync_ui_from_stop()

    # ---------------------------------------------------------------
    # Event handlers
    # ---------------------------------------------------------------

    def _hit_test_handle(self, local_x: float, local_y: float) -> Optional[int]:
        """Test if click hits a handle, return stop index."""
        handle_base = self.gradient_bar_bottom + 4
        handle_top = handle_base
        handle_bottom = handle_base + 16

        for idx, stop in enumerate(self.stops):
            x = self._stop_to_x(stop["pos"])
            if abs(local_x - x) <= self.handle_radius * 1.2 and handle_top <= local_y <= handle_bottom:
                return idx
        return None

    def _gradient_local_coords(self, mouse_x: float, mouse_y: float) -> Optional[tuple[float, float]]:
        """Get local coordinates within gradient drawlist."""
        if not dpg.does_item_exist(self.gradient_drawlist_id):
            return None
        rect_min = dpg.get_item_rect_min(self.gradient_drawlist_id)
        rect_max = dpg.get_item_rect_max(self.gradient_drawlist_id)
        if rect_min is None or rect_max is None:
            return None
        if not (rect_min[0] <= mouse_x <= rect_max[0] and rect_min[1] <= mouse_y <= rect_max[1]):
            return None
        return mouse_x - rect_min[0], mouse_y - rect_min[1]

    def _on_gradient_mouse_down(self, sender, app_data):
        """Handle mouse down on gradient bar."""
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        coords = self._gradient_local_coords(mouse_x, mouse_y)
        if coords is None:
            return

        local_x, local_y = coords
        handle_idx = self._hit_test_handle(local_x, local_y)

        if handle_idx is not None:
            # Select and start dragging
            self.selected_stop_id = self.stops[handle_idx]["id"]
            self.is_dragging = True
            self._refresh_all()
        elif self.gradient_bar_top <= local_y <= self.gradient_bar_bottom:
            # Add new stop
            pos = self._x_to_pos(local_x)
            L, C, H = self._interpolate_oklch(pos)
            self._add_stop(pos, L, C, H)
            self.is_dragging = True

    def _on_gradient_mouse_drag(self, sender, app_data):
        """Handle drag on gradient bar."""
        if not self.is_dragging or self.selected_stop_id is None:
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        coords = self._gradient_local_coords(mouse_x, mouse_y)
        if coords is None:
            return

        local_x, _ = coords
        new_pos = self._x_to_pos(local_x)

        idx = self._get_stop_index(self.selected_stop_id)
        if idx is not None:
            # Clamp to neighbors
            min_pos = self.stops[idx - 1]["pos"] + 0.001 if idx > 0 else 0.0
            max_pos = self.stops[idx + 1]["pos"] - 0.001 if idx < len(self.stops) - 1 else 1.0
            self.stops[idx]["pos"] = float(np.clip(new_pos, min_pos, max_pos))

            self._update_gradient_texture()
            self._update_gradient_drawlist()

    def _on_gradient_mouse_release(self, sender, app_data):
        """Handle mouse release."""
        if self.is_dragging:
            self.is_dragging = False
            self._sort_stops()
            self._refresh_all()

    def _on_gradient_double_click(self, sender, app_data):
        """Handle double-click to delete stop."""
        if self.is_dragging:
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        coords = self._gradient_local_coords(mouse_x, mouse_y)
        if coords is None:
            return

        local_x, local_y = coords
        handle_idx = self._hit_test_handle(local_x, local_y)
        if handle_idx is not None and len(self.stops) > 2:
            stop_id = self.stops[handle_idx]["id"]
            self._remove_stop(stop_id)

    def _on_slice_click(self, sender, app_data):
        """Handle click on C x H slice."""
        if not dpg.does_item_exist(self.slice_drawlist_id):
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min(self.slice_drawlist_id)
        rect_max = dpg.get_item_rect_max(self.slice_drawlist_id)
        if rect_min is None or rect_max is None:
            return

        if not (rect_min[0] <= mouse_x <= rect_max[0] and rect_min[1] <= mouse_y <= rect_max[1]):
            return

        local_x = mouse_x - rect_min[0]
        local_y = mouse_y - rect_min[1]

        stop = self._get_selected_stop()
        if stop is None:
            return

        # Convert to H, C
        new_h = float(np.clip(local_x, 0, 359.9))
        new_c = float(np.clip((local_y / self.slice_height) * self.c_max_display, 0, self.c_max_display))

        stop["H"] = new_h
        stop["C"] = new_c

        dpg.set_value("h_slider", new_h)
        dpg.set_value("c_slider", new_c)

        self._update_gradient_texture()
        self._update_l_gradient_texture()
        self._update_preview_texture()
        self._update_gradient_drawlist()
        self._update_slice_crosshair()
        self._sync_ui_from_stop()

    def _on_l_change(self, sender, value):
        """Handle L slider change."""
        stop = self._get_selected_stop()
        if stop is None:
            return
        stop["L"] = value
        self._refresh_all()

    def _on_c_change(self, sender, value):
        """Handle C slider change."""
        stop = self._get_selected_stop()
        if stop is None:
            return
        stop["C"] = value
        self._update_gradient_texture()
        self._update_l_gradient_texture()
        self._update_preview_texture()
        self._update_gradient_drawlist()
        self._update_slice_crosshair()
        self._sync_ui_from_stop()

    def _on_h_change(self, sender, value):
        """Handle H slider change."""
        stop = self._get_selected_stop()
        if stop is None:
            return
        stop["H"] = value
        self._update_gradient_texture()
        self._update_l_gradient_texture()
        self._update_preview_texture()
        self._update_gradient_drawlist()
        self._update_slice_crosshair()
        self._sync_ui_from_stop()

    # ---------------------------------------------------------------
    # Stop management
    # ---------------------------------------------------------------

    def _add_stop(self, pos: float, L: float, C: float, H: float) -> int:
        """Add a new stop."""
        stop = {
            "id": self._next_stop_id,
            "pos": pos,
            "L": L,
            "C": C,
            "H": H,
        }
        self._next_stop_id += 1
        self.stops.append(stop)
        self._sort_stops()
        self.selected_stop_id = stop["id"]
        self._refresh_all()
        return stop["id"]

    def _remove_stop(self, stop_id: int):
        """Remove a stop."""
        if len(self.stops) <= 2:
            return

        idx = self._get_stop_index(stop_id)
        if idx is None:
            return

        del self.stops[idx]

        # Select neighboring stop
        if self.stops:
            new_idx = min(idx, len(self.stops) - 1)
            self.selected_stop_id = self.stops[new_idx]["id"]
        else:
            self.selected_stop_id = None

        self._refresh_all()

    def _on_delete_stop(self):
        """Delete selected stop."""
        if self.selected_stop_id is not None:
            self._remove_stop(self.selected_stop_id)

    def _on_add_stop(self):
        """Add stop at midpoint."""
        L, C, H = self._interpolate_oklch(0.5)
        self._add_stop(0.5, L, C, H)

    def _on_clamp_to_gamut(self):
        """Clamp selected stop's chroma to gamut."""
        stop = self._get_selected_stop()
        if stop is None:
            return
        max_c = self._get_max_chroma(stop["L"], stop["H"])
        if stop["C"] > max_c:
            stop["C"] = max_c
            dpg.set_value("c_slider", max_c)
            self._refresh_all()

    # ---------------------------------------------------------------
    # Save/Load
    # ---------------------------------------------------------------

    def _on_save_palette(self):
        """Save current palette to library."""
        name = dpg.get_value("palette_name_input").strip()
        if not name:
            print("Enter a name first")
            return

        # Save OKLCH stops
        stops_data = [
            {"pos": s["pos"], "L": s["L"], "C": s["C"], "H": s["H"]}
            for s in self.stops
        ]
        self.saved_palettes[name] = stops_data
        self._save_palettes()

        # Also add to runtime library as RGB LUT
        lut = self._build_lut(size=16)
        add_palette(name, lut.tolist())

        print(f"Saved palette '{name}'")

    def _on_load_palette(self, sender, app_data, user_data):
        """Load a saved palette."""
        name = user_data
        if name not in self.saved_palettes:
            return

        stops_data = self.saved_palettes[name]
        self.stops.clear()
        self._next_stop_id = 0

        for s in stops_data:
            self.stops.append({
                "id": self._next_stop_id,
                "pos": s["pos"],
                "L": s["L"],
                "C": s["C"],
                "H": s["H"],
            })
            self._next_stop_id += 1

        self._sort_stops()
        self.selected_stop_id = self.stops[0]["id"] if self.stops else None
        dpg.set_value("palette_name_input", name)
        self._refresh_all()

    def _on_new_palette(self):
        """Reset to default gradient."""
        self._init_default_gradient()
        dpg.set_value("palette_name_input", "")
        self._refresh_all()

    # ---------------------------------------------------------------
    # Presets
    # ---------------------------------------------------------------

    def _load_preset(self, preset_name: str):
        """Load a built-in preset."""
        presets = {
            "Grayscale": [
                {"pos": 0.0, "L": 0.0, "C": 0.0, "H": 0.0},
                {"pos": 1.0, "L": 1.0, "C": 0.0, "H": 0.0},
            ],
            "Warm": [
                {"pos": 0.0, "L": 0.15, "C": 0.08, "H": 30.0},
                {"pos": 0.5, "L": 0.65, "C": 0.15, "H": 50.0},
                {"pos": 1.0, "L": 0.95, "C": 0.10, "H": 80.0},
            ],
            "Cool": [
                {"pos": 0.0, "L": 0.10, "C": 0.10, "H": 250.0},
                {"pos": 0.5, "L": 0.55, "C": 0.12, "H": 220.0},
                {"pos": 1.0, "L": 0.95, "C": 0.05, "H": 200.0},
            ],
            "Sunset": [
                {"pos": 0.0, "L": 0.15, "C": 0.12, "H": 300.0},
                {"pos": 0.33, "L": 0.50, "C": 0.20, "H": 30.0},
                {"pos": 0.66, "L": 0.70, "C": 0.18, "H": 60.0},
                {"pos": 1.0, "L": 0.95, "C": 0.08, "H": 90.0},
            ],
            "Ocean": [
                {"pos": 0.0, "L": 0.10, "C": 0.10, "H": 240.0},
                {"pos": 0.5, "L": 0.50, "C": 0.15, "H": 200.0},
                {"pos": 1.0, "L": 0.90, "C": 0.08, "H": 180.0},
            ],
        }

        if preset_name not in presets:
            return

        self.stops.clear()
        self._next_stop_id = 0
        for s in presets[preset_name]:
            self.stops.append({
                "id": self._next_stop_id,
                "pos": s["pos"],
                "L": s["L"],
                "C": s["C"],
                "H": s["H"],
            })
            self._next_stop_id += 1

        self._sort_stops()
        self.selected_stop_id = self.stops[0]["id"] if self.stops else None
        self._refresh_all()

    # ---------------------------------------------------------------
    # UI build
    # ---------------------------------------------------------------

    def _build_ui(self):
        """Build the DearPyGui interface."""
        dpg.create_context()

        self.texture_registry = dpg.add_texture_registry()

        # Initialize textures
        self._update_gradient_texture()
        self._update_slice_texture()
        self._update_l_gradient_texture()
        self._update_preview_texture()

        # Mouse handlers
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=self._on_gradient_mouse_down)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,
                                       callback=self._on_gradient_mouse_drag, threshold=0.0)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                          callback=self._on_gradient_mouse_release)
            dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Left,
                                               callback=self._on_gradient_double_click)

        # Slice click handler
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=self._on_slice_click)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,
                                       callback=self._on_slice_click, threshold=0.0)

        # Main window
        with dpg.window(label="OKLCH Palette Creator", tag="main_window"):
            dpg.add_text("OKLCH Palette Creator", color=(150, 200, 255))
            dpg.add_text("Click gradient bar to add stops, drag to move, double-click to delete",
                        color=(150, 150, 150))
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Gradient bar
            dpg.add_text("Gradient Preview")
            self.gradient_drawlist_id = dpg.add_drawlist(
                width=self.gradient_width,
                height=self.gradient_height,
            )
            self._update_gradient_drawlist()

            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                # Left: OKLCH picker
                with dpg.group():
                    dpg.add_text("Stop Color (C x H slice at current L)")
                    self.slice_drawlist_id = dpg.add_drawlist(
                        width=self.slice_width,
                        height=self.slice_height,
                    )
                    self._update_slice_crosshair()

                    dpg.add_spacer(height=10)

                    # L slider with gradient
                    dpg.add_text("Lightness (L)")
                    dpg.add_image(self.l_gradient_texture_id, width=360, height=16)
                    dpg.add_slider_float(
                        label="",
                        default_value=0.5,
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
                        default_value=0.0,
                        min_value=0.0,
                        max_value=self.c_max_display,
                        format="%.4f",
                        width=360,
                        callback=self._on_c_change,
                        tag="c_slider",
                    )

                    dpg.add_spacer(height=8)

                    # H slider
                    dpg.add_text("Hue (H)")
                    dpg.add_slider_float(
                        label="",
                        default_value=0.0,
                        min_value=0.0,
                        max_value=360.0,
                        format="%.1f",
                        width=360,
                        callback=self._on_h_change,
                        tag="h_slider",
                    )

                dpg.add_spacer(width=30)

                # Right: Info and controls
                with dpg.group():
                    dpg.add_text("Preview")
                    dpg.add_image(self.preview_texture_id, width=60, height=60)

                    dpg.add_spacer(height=10)
                    dpg.add_text("", tag="stop_info_text")

                    dpg.add_spacer(height=10)
                    dpg.add_button(label="Clamp to Gamut", width=140,
                                  callback=lambda: self._on_clamp_to_gamut())
                    dpg.add_button(label="Delete Stop", width=140,
                                  callback=lambda: self._on_delete_stop())
                    dpg.add_button(label="Add Stop at 0.5", width=140,
                                  callback=lambda: self._on_add_stop())

                    dpg.add_spacer(height=20)
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    dpg.add_text("Save / Load")
                    dpg.add_input_text(label="Name", default_value="", width=140, tag="palette_name_input")
                    dpg.add_button(label="Save Palette", width=140,
                                  callback=lambda: self._on_save_palette())
                    dpg.add_button(label="New Palette", width=140,
                                  callback=lambda: self._on_new_palette())

                    dpg.add_spacer(height=10)
                    dpg.add_text("Saved Palettes:", color=(150, 150, 150))
                    for name in sorted(self.saved_palettes.keys()):
                        dpg.add_button(label=name, width=140,
                                      callback=self._on_load_palette, user_data=name)

                    dpg.add_spacer(height=20)
                    dpg.add_separator()
                    dpg.add_spacer(height=10)

                    dpg.add_text("Presets:", color=(150, 150, 150))
                    for preset in ["Grayscale", "Warm", "Cool", "Sunset", "Ocean"]:
                        dpg.add_button(label=preset, width=140,
                                      callback=lambda s, a, u=preset: self._load_preset(u))

        # Viewport
        dpg.create_viewport(title="OKLCH Palette Creator", width=900, height=700)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

        # Initial sync
        self._sync_ui_from_stop()

    def run(self):
        """Launch the palette creator."""
        self._build_ui()
        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    creator = OklchPaletteCreator()
    creator.run()


if __name__ == "__main__":
    main()
