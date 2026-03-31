"""Palette editor functionality mixin for PostprocessingPanel.

This mixin provides the inline OKLCH color palette editor including:
- Gradient stop manipulation (drag, add, remove)
- Chroma/Hue slice picker
- Lightness gradient with slider
- Palette preview and selection
- Delete confirmation modal
"""

import time
from typing import Optional, TYPE_CHECKING

import numpy as np

from elliptica.app.state_manager import StateKey
from elliptica.colorspace import (
    build_oklch_lut,
    gamut_map_to_srgb,
    interpolate_oklch,
    max_chroma_fast,
    srgb_to_oklch,
)
from elliptica.palettes import (
    delete_palette,
    get_palette_spec,
    list_color_palettes,
    set_palette_spec,
)

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class PaletteEditorMixin:
    """Mixin providing palette editor functionality for PostprocessingPanel.

    This mixin expects the following attributes to be present on the class:
        - app: EllipticaApp instance
        - palette_preview_width: int
        - palette_preview_height: int
        - update_context_ui(): method to refresh UI state
        - _get_current_region_style_unlocked(): method returning RegionStyle
    """

    # Type hints for attributes expected from the main class
    app: "EllipticaApp"
    palette_preview_width: int
    palette_preview_height: int

    def _init_palette_editor_state(self) -> None:
        """Initialize palette editor-related instance variables.

        Call this from the main class's __init__ method.
        """
        # Palette editor UI state
        self.palette_editor_active: bool = False
        self.palette_editor_palette_name: Optional[str] = None
        self.palette_editor_for_region: bool = False
        self.palette_editor_group_id: Optional[int] = None
        self.palette_editor_title_id: Optional[int] = None
        self.palette_editor_gradient_drawlist_id: Optional[int] = None
        self.palette_editor_slice_drawlist_id: Optional[int] = None
        self.palette_editor_l_gradient_drawlist_id: Optional[int] = None
        self.palette_editor_interp_mode_id: Optional[int] = None
        self.palette_editor_l_slider_id: Optional[int] = None
        self.palette_editor_c_slider_id: Optional[int] = None
        self.palette_editor_h_slider_id: Optional[int] = None

        # Palette editor dimensions
        self.palette_editor_width = self.palette_preview_width
        self.palette_editor_gradient_height = 60
        self.palette_editor_slice_height = 110
        self.palette_editor_lbar_height = 16
        self.palette_editor_lbar_width = self.palette_editor_width

        # Texture IDs
        self.palette_editor_gradient_texture_id: Optional[int] = None
        self.palette_editor_slice_texture_id: Optional[int] = None
        self.palette_editor_l_gradient_texture_id: Optional[int] = None
        self.palette_editor_handler_registry_id: Optional[int] = None

        # Stop management
        self.palette_editor_stops: list[dict] = []
        self.palette_editor_next_stop_id: int = 0
        self.palette_editor_selected_stop_id: Optional[int] = None
        self.palette_editor_dragging_stop_id: Optional[int] = None
        self.palette_editor_is_dragging: bool = False
        self.palette_editor_slice_drag_active: bool = False

        # Editor parameters
        self.palette_editor_relative_chroma: bool = True
        self.palette_editor_interp_mix: float = 1.0
        self.palette_editor_c_max_absolute: float = 0.5
        self.palette_editor_c_max_display: float = 1.0
        self.palette_editor_max_stops: int = 12

        # Geometry constants
        self.palette_editor_gradient_padding = 14
        self.palette_editor_gradient_bar_top = 6
        self.palette_editor_gradient_bar_bottom = self.palette_editor_gradient_height - 18
        self.palette_editor_handle_radius = 7
        self.palette_editor_handle_hit_radius = 12
        self.palette_editor_handle_center_y = self.palette_editor_gradient_bar_bottom

        # Precomputed grids for slice
        self.palette_editor_h_grid = np.array([], dtype=np.float32)
        self.palette_editor_c_grid = np.array([], dtype=np.float32)
        self.palette_editor_h_mesh = np.array([], dtype=np.float32)
        self.palette_editor_c_mesh = np.array([], dtype=np.float32)

        # Refresh throttling
        self.palette_editor_refresh_pending: bool = False
        self.palette_editor_last_refresh_time: float = 0.0
        self.palette_editor_refresh_throttle: float = 0.1

        # Dirty flags
        self.palette_editor_dirty: bool = False
        self.palette_editor_persist_dirty: bool = False
        self.palette_editor_needs_colormap_rebuild: bool = False
        self.palette_editor_syncing: bool = False

        # Delete confirmation modal
        self.delete_confirmation_modal_id: Optional[int] = None
        self.pending_delete_palette: Optional[str] = None

        # Palette preview button IDs
        self.global_palette_preview_button_id: Optional[int] = None
        self.region_palette_preview_button_id: Optional[int] = None

        # Initialize grids
        self._rebuild_palette_editor_grids()

    # ------------------------------------------------------------------
    # UI Building
    # ------------------------------------------------------------------

    def _build_palette_editor_ui(self, parent) -> None:
        """Build the inline palette editor shell."""
        if dpg is None:
            return

        with dpg.group(tag="palette_editor_group", show=False, parent=parent) as group:
            self.palette_editor_group_id = group

            with dpg.group(horizontal=True):
                self.palette_editor_title_id = dpg.add_text(
                    "Editing: (none)",
                    tag="palette_editor_title",
                    color=(150, 200, 255),
                )
                dpg.add_spacer(width=8)
                dpg.add_button(
                    label="Done",
                    width=60,
                    callback=self.on_palette_editor_done,
                    tag="palette_editor_done_btn",
                )

            dpg.add_spacer(height=6)
            dpg.add_separator()
            dpg.add_spacer(height=6)

            dpg.add_text("Gradient", color=(150, 150, 150))
            self.palette_editor_gradient_drawlist_id = dpg.add_drawlist(
                width=self.palette_editor_width,
                height=self.palette_editor_gradient_height,
                tag="palette_editor_gradient_drawlist",
            )

            dpg.add_spacer(height=6)
            dpg.add_text("Chroma / Hue (C x H)", color=(150, 150, 150))
            self.palette_editor_slice_drawlist_id = dpg.add_drawlist(
                width=self.palette_editor_width,
                height=self.palette_editor_slice_height,
                tag="palette_editor_slice_drawlist",
            )

            dpg.add_spacer(height=6)
            dpg.add_text("Lightness (L)", color=(150, 150, 150))
            slider_width = max(140, self.palette_editor_width - 60)
            self.palette_editor_lbar_width = slider_width
            self.palette_editor_l_gradient_drawlist_id = dpg.add_drawlist(
                width=self.palette_editor_lbar_width,
                height=self.palette_editor_lbar_height,
                tag="palette_editor_l_gradient_drawlist",
            )

            dpg.add_spacer(height=4)
            self.palette_editor_l_slider_id = dpg.add_slider_float(
                label="L",
                default_value=0.5,
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
                width=slider_width,
                enabled=False,
                callback=self.on_palette_editor_l_slider,
                tag="palette_editor_l_slider",
            )
            self.palette_editor_c_slider_id = dpg.add_slider_float(
                label="C",
                default_value=0.0,
                min_value=0.0,
                max_value=0.4,
                format="%.4f",
                width=slider_width,
                enabled=False,
                callback=self.on_palette_editor_c_slider,
                tag="palette_editor_c_slider",
            )
            self.palette_editor_h_slider_id = dpg.add_slider_float(
                label="H",
                default_value=0.0,
                min_value=0.0,
                max_value=360.0,
                format="%.1f",
                width=slider_width,
                enabled=False,
                callback=self.on_palette_editor_h_slider,
                tag="palette_editor_h_slider",
            )

            dpg.add_spacer(height=6)
            dpg.add_text("Chroma interpolation", color=(150, 150, 150))
            self.palette_editor_interp_mode_id = dpg.add_radio_button(
                items=["Relative", "Absolute"],
                default_value="Relative",
                horizontal=True,
                callback=self.on_palette_editor_interp_mode,
                tag="palette_editor_interp_mode",
            )

            dpg.add_spacer(height=6)

        self._draw_palette_editor_placeholders()

    def _draw_palette_editor_placeholders(self) -> None:
        """Draw placeholder panels for the palette editor."""
        if dpg is None:
            return

        def _draw_panel(drawlist_id: Optional[int], width: int, height: int) -> None:
            if drawlist_id is None or not dpg.does_item_exist(drawlist_id):
                return
            dpg.delete_item(drawlist_id, children_only=True)
            dpg.draw_rectangle(
                (0, 0),
                (width, height),
                color=(70, 70, 70, 255),
                fill=(25, 25, 25, 255),
                thickness=1,
                parent=drawlist_id,
            )

        _draw_panel(self.palette_editor_gradient_drawlist_id, self.palette_editor_width, self.palette_editor_gradient_height)
        _draw_panel(self.palette_editor_slice_drawlist_id, self.palette_editor_width, self.palette_editor_slice_height)
        _draw_panel(self.palette_editor_l_gradient_drawlist_id, self.palette_editor_width, self.palette_editor_lbar_height)

    # ------------------------------------------------------------------
    # Grid and handler setup
    # ------------------------------------------------------------------

    def _rebuild_palette_editor_grids(self) -> None:
        """Rebuild precomputed grids for the CxH slice."""
        self.palette_editor_c_max_display = 1.0 if self.palette_editor_relative_chroma else self.palette_editor_c_max_absolute
        self.palette_editor_h_grid = np.linspace(
            0.0,
            360.0,
            max(1, int(self.palette_editor_width)),
            endpoint=False,
            dtype=np.float32,
        )
        self.palette_editor_c_grid = np.linspace(
            0.0,
            self.palette_editor_c_max_display,
            max(1, int(self.palette_editor_slice_height)),
            dtype=np.float32,
        )
        self.palette_editor_h_mesh, self.palette_editor_c_mesh = np.meshgrid(
            self.palette_editor_h_grid,
            self.palette_editor_c_grid,
        )

    def _ensure_palette_editor_handlers(self) -> None:
        if dpg is None or self.palette_editor_handler_registry_id is not None:
            return

        with dpg.handler_registry() as handler_reg:
            self.palette_editor_handler_registry_id = handler_reg
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._on_palette_editor_mouse_down,
            )
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._on_palette_editor_mouse_drag,
                threshold=0.0,
            )
            dpg.add_mouse_release_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._on_palette_editor_mouse_release,
            )
            dpg.add_mouse_double_click_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._on_palette_editor_mouse_double_click,
            )

    # ------------------------------------------------------------------
    # Control enable/disable
    # ------------------------------------------------------------------

    def _set_palette_editor_controls_enabled(self, enabled: bool) -> None:
        if dpg is None:
            return

        if self.palette_editor_l_slider_id is not None and dpg.does_item_exist(self.palette_editor_l_slider_id):
            dpg.configure_item(self.palette_editor_l_slider_id, enabled=enabled)
        if self.palette_editor_c_slider_id is not None and dpg.does_item_exist(self.palette_editor_c_slider_id):
            dpg.configure_item(self.palette_editor_c_slider_id, enabled=enabled)
        if self.palette_editor_h_slider_id is not None and dpg.does_item_exist(self.palette_editor_h_slider_id):
            dpg.configure_item(self.palette_editor_h_slider_id, enabled=enabled)

    def _palette_editor_item_active(self, item_id: Optional[int]) -> bool:
        if dpg is None or item_id is None:
            return False
        if not dpg.does_item_exist(item_id):
            return False
        return bool(dpg.is_item_active(item_id))

    def _palette_editor_any_slider_active(self) -> bool:
        return (
            self._palette_editor_item_active(self.palette_editor_l_slider_id)
            or self._palette_editor_item_active(self.palette_editor_c_slider_id)
            or self._palette_editor_item_active(self.palette_editor_h_slider_id)
        )

    # ------------------------------------------------------------------
    # Stop management
    # ------------------------------------------------------------------

    def _palette_editor_get_stop_index(self, stop_id: int) -> Optional[int]:
        for idx, stop in enumerate(self.palette_editor_stops):
            if stop["id"] == stop_id:
                return idx
        return None

    def _palette_editor_get_selected_stop(self) -> Optional[dict]:
        if self.palette_editor_selected_stop_id is None:
            return None
        idx = self._palette_editor_get_stop_index(self.palette_editor_selected_stop_id)
        if idx is None:
            return None
        return self.palette_editor_stops[idx]

    def _palette_editor_sort_stops(self) -> None:
        self.palette_editor_stops.sort(key=lambda s: s["pos"])

    def _palette_editor_stop_to_x(self, pos: float) -> float:
        bar_left = self.palette_editor_gradient_padding
        bar_right = self.palette_editor_width - self.palette_editor_gradient_padding
        if bar_right <= bar_left:
            return bar_left
        return bar_left + pos * (bar_right - bar_left)

    def _palette_editor_x_to_pos(self, x: float) -> float:
        bar_left = self.palette_editor_gradient_padding
        bar_right = self.palette_editor_width - self.palette_editor_gradient_padding
        if bar_right <= bar_left:
            return 0.0
        return float(np.clip((x - bar_left) / (bar_right - bar_left), 0.0, 1.0))

    def _palette_editor_drawlist_local_coords(
        self,
        drawlist_id: Optional[int],
        mouse_x: float,
        mouse_y: float,
        clamp: bool = False,
    ) -> Optional[tuple[float, float]]:
        if dpg is None or drawlist_id is None or not dpg.does_item_exist(drawlist_id):
            return None

        rect_min = dpg.get_item_rect_min(drawlist_id)
        rect_max = dpg.get_item_rect_max(drawlist_id)

        inside = rect_min[0] <= mouse_x <= rect_max[0] and rect_min[1] <= mouse_y <= rect_max[1]
        if not inside and not clamp:
            return None

        local_x = mouse_x - rect_min[0]
        local_y = mouse_y - rect_min[1]

        if clamp:
            local_x = float(np.clip(local_x, 0.0, rect_max[0] - rect_min[0]))
            local_y = float(np.clip(local_y, 0.0, rect_max[1] - rect_min[1]))

        return float(local_x), float(local_y)

    def _palette_editor_hit_test_handle(self, local_x: float, local_y: float) -> Optional[int]:
        best_idx = None
        best_dist = float(self.palette_editor_handle_hit_radius)

        for idx, stop in enumerate(self.palette_editor_stops):
            x = self._palette_editor_stop_to_x(stop["pos"])
            y = self.palette_editor_handle_center_y
            dist = float(np.hypot(local_x - x, local_y - y))
            if dist <= self.palette_editor_handle_hit_radius and dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx

    def _palette_editor_add_stop(self, pos: float) -> None:
        L, C, H = interpolate_oklch(
            self.palette_editor_stops,
            pos,
            relative_chroma=self.palette_editor_relative_chroma,
            interp_mix=self.palette_editor_interp_mix,
        )
        new_id = self.palette_editor_next_stop_id
        self.palette_editor_next_stop_id += 1
        self.palette_editor_stops.append({
            "id": new_id,
            "pos": float(np.clip(pos, 0.0, 1.0)),
            "L": float(L),
            "C": float(C),
            "H": float(H),
        })
        self.palette_editor_selected_stop_id = new_id
        self._palette_editor_sort_stops()

    def _palette_editor_remove_stop(self, stop_id: int) -> None:
        if len(self.palette_editor_stops) <= 2:
            return
        idx = self._palette_editor_get_stop_index(stop_id)
        if idx is None:
            return
        del self.palette_editor_stops[idx]
        if self.palette_editor_stops:
            new_idx = min(idx, len(self.palette_editor_stops) - 1)
            self.palette_editor_selected_stop_id = self.palette_editor_stops[new_idx]["id"]
        else:
            self.palette_editor_selected_stop_id = None

    # ------------------------------------------------------------------
    # Chroma conversion utilities
    # ------------------------------------------------------------------

    def _palette_editor_chroma_rel_to_abs(self, L: float, H: float, C_rel: float) -> float:
        max_c = max_chroma_fast(
            np.array(L, dtype=np.float32),
            np.array(H, dtype=np.float32),
        )
        max_c_val = float(max_c)
        if max_c_val <= 0.0:
            return 0.0
        return float(np.clip(C_rel, 0.0, 1.0) * max_c_val)

    def _palette_editor_chroma_abs_to_rel(self, L: float, H: float, C_abs: float) -> float:
        max_c = max_chroma_fast(
            np.array(L, dtype=np.float32),
            np.array(H, dtype=np.float32),
        )
        max_c_val = float(max_c)
        if max_c_val <= 0.0:
            return 0.0
        return float(np.clip(C_abs / max_c_val, 0.0, 1.0))

    def _palette_editor_chroma_rel_to_abs_array(
        self,
        L_arr: np.ndarray,
        H_arr: np.ndarray,
        C_rel_arr: np.ndarray,
    ) -> np.ndarray:
        max_c = max_chroma_fast(L_arr, H_arr)
        return np.clip(C_rel_arr, 0.0, 1.0) * max_c

    def _palette_editor_chroma_abs_to_rel_array(
        self,
        L_arr: np.ndarray,
        H_arr: np.ndarray,
        C_abs_arr: np.ndarray,
    ) -> np.ndarray:
        max_c = max_chroma_fast(L_arr, H_arr)
        safe = np.where(max_c <= 0.0, 0.0, C_abs_arr / max_c)
        return np.clip(safe, 0.0, 1.0)

    def _palette_editor_oklch_to_rgb(self, L: float, C: float, H: float) -> tuple[float, float, float]:
        if self.palette_editor_relative_chroma:
            C_abs = self._palette_editor_chroma_rel_to_abs(L, H, C)
        else:
            C_abs = float(max(C, 0.0))
        rgb = gamut_map_to_srgb(
            np.array([L], dtype=np.float32),
            np.array([C_abs], dtype=np.float32),
            np.array([H], dtype=np.float32),
            method="compress",
        )
        rgb = np.clip(np.array(rgb, dtype=np.float32), 0.0, 1.0)
        return float(rgb[0, 0]), float(rgb[0, 1]), float(rgb[0, 2])

    def _palette_editor_oklch_to_rgb255(self, L: float, C: float, H: float) -> tuple[int, int, int]:
        r, g, b = self._palette_editor_oklch_to_rgb(L, C, H)
        return int(r * 255), int(g * 255), int(b * 255)

    def _palette_editor_is_in_gamut(self, L: float, C: float, H: float) -> bool:
        if self.palette_editor_relative_chroma:
            return C <= 1.0 + 1e-6
        max_c = max_chroma_fast(
            np.array(L, dtype=np.float32),
            np.array(H, dtype=np.float32),
        )
        return C <= float(max_c) + 1e-6

    def _palette_editor_update_c_slider_range(self) -> None:
        if dpg is None or self.palette_editor_c_slider_id is None:
            return
        if not dpg.does_item_exist(self.palette_editor_c_slider_id):
            return
        max_value = 1.0 if self.palette_editor_relative_chroma else self.palette_editor_c_max_absolute
        fmt = "%.3f" if self.palette_editor_relative_chroma else "%.4f"
        dpg.configure_item(self.palette_editor_c_slider_id, max_value=max_value, format=fmt)

    # ------------------------------------------------------------------
    # Default and conversion
    # ------------------------------------------------------------------

    def _palette_editor_default_stops(self) -> list[dict]:
        c0 = 0.08
        c1 = 0.06
        if self.palette_editor_relative_chroma:
            c0 = self._palette_editor_chroma_abs_to_rel(0.20, 270.0, c0)
            c1 = self._palette_editor_chroma_abs_to_rel(0.92, 60.0, c1)
        return [
            {"id": 0, "pos": 0.0, "L": 0.20, "C": c0, "H": 270.0},
            {"id": 1, "pos": 1.0, "L": 0.92, "C": c1, "H": 60.0},
        ]

    def _palette_editor_convert_rgb_stops(self, stops: list[dict]) -> list[dict]:
        if not stops:
            return []

        positions = []
        colors = []
        missing_pos = False
        for stop in stops:
            if "pos" not in stop:
                missing_pos = True
                break
            positions.append(float(stop["pos"]))
            colors.append([
                float(stop["r"]),
                float(stop["g"]),
                float(stop["b"]),
            ])

        if missing_pos:
            count = len(stops)
            if count == 1:
                positions = [0.0]
            else:
                positions = list(np.linspace(0.0, 1.0, count, dtype=np.float32))
            colors = [
                [float(stop["r"]), float(stop["g"]), float(stop["b"])]
                for stop in stops
            ]

        positions_arr = np.array(positions, dtype=np.float32)
        colors_arr = np.array(colors, dtype=np.float32)
        if positions_arr.size == 0 or colors_arr.size == 0:
            return []

        order = np.argsort(positions_arr)
        positions_arr = positions_arr[order]
        colors_arr = colors_arr[order]
        positions_arr = np.clip(positions_arr, 0.0, 1.0)

        if colors_arr.shape[0] > self.palette_editor_max_stops:
            if positions_arr[0] > 0.0:
                positions_arr = np.insert(positions_arr, 0, 0.0)
                colors_arr = np.vstack([colors_arr[0], colors_arr])
            if positions_arr[-1] < 1.0:
                positions_arr = np.append(positions_arr, 1.0)
                colors_arr = np.vstack([colors_arr, colors_arr[-1]])

            sample_positions = np.linspace(
                0.0,
                1.0,
                self.palette_editor_max_stops,
                dtype=np.float32,
            )
            sampled = np.stack(
                [
                    np.interp(sample_positions, positions_arr, colors_arr[:, 0]),
                    np.interp(sample_positions, positions_arr, colors_arr[:, 1]),
                    np.interp(sample_positions, positions_arr, colors_arr[:, 2]),
                ],
                axis=1,
            )
            positions_arr = sample_positions
            colors_arr = sampled

        rgb = np.clip(colors_arr, 0.0, 1.0)
        L, C, H = srgb_to_oklch(rgb)
        C_rel = self._palette_editor_chroma_abs_to_rel_array(L, H, C)

        converted = []
        for idx, pos in enumerate(positions_arr):
            converted.append({
                "id": idx,
                "pos": float(pos),
                "L": float(L[idx]),
                "C": float(C_rel[idx]),
                "H": float(H[idx]),
            })
        return converted

    # ------------------------------------------------------------------
    # Palette loading and saving
    # ------------------------------------------------------------------

    def _palette_editor_load_palette(self, palette_name: str) -> None:
        spec = get_palette_spec(palette_name)
        if spec is None:
            return

        space = spec.get("space", "rgb")
        stops = spec.get("stops", [])
        relative_chroma = bool(spec.get("relative_chroma", True))
        interp_mix = float(spec.get("interp_mix", 1.0))
        interp_mix = 1.0 if interp_mix >= 0.5 else 0.0

        self.palette_editor_relative_chroma = relative_chroma
        self.palette_editor_interp_mix = interp_mix

        if space == "oklch":
            editor_stops: list[dict] = []
            for idx, stop in enumerate(stops):
                try:
                    editor_stops.append({
                        "id": idx,
                        "pos": float(stop["pos"]),
                        "L": float(stop["L"]),
                        "C": float(stop["C"]),
                        "H": float(stop["H"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            # Convert RGB palettes to OKLCH for editing (relative chroma).
            self.palette_editor_relative_chroma = True
            self.palette_editor_interp_mix = 1.0
            editor_stops = self._palette_editor_convert_rgb_stops(stops)

        if len(editor_stops) < 2:
            editor_stops = self._palette_editor_default_stops()

        editor_stops.sort(key=lambda s: s["pos"])

        self.palette_editor_stops = editor_stops
        self.palette_editor_next_stop_id = max(stop["id"] for stop in editor_stops) + 1
        self.palette_editor_selected_stop_id = editor_stops[0]["id"]
        self.palette_editor_dragging_stop_id = None
        self.palette_editor_is_dragging = False
        self.palette_editor_slice_drag_active = False
        self.palette_editor_refresh_pending = False
        self.palette_editor_dirty = False
        self.palette_editor_persist_dirty = False
        self.palette_editor_needs_colormap_rebuild = False

        self._rebuild_palette_editor_grids()
        self._palette_editor_update_c_slider_range()
        self._ensure_palette_editor_handlers()
        self._palette_editor_refresh_visuals()
        self._palette_editor_sync_interp_mode()

    def _palette_editor_interp_label(self) -> str:
        return "Relative" if self.palette_editor_interp_mix >= 0.5 else "Absolute"

    def _palette_editor_sync_interp_mode(self) -> None:
        if dpg is None or self.palette_editor_interp_mode_id is None:
            return
        if not dpg.does_item_exist(self.palette_editor_interp_mode_id):
            return
        dpg.set_value(self.palette_editor_interp_mode_id, self._palette_editor_interp_label())

    def _palette_editor_build_spec(self) -> dict:
        stops = sorted(self.palette_editor_stops, key=lambda s: s["pos"])
        spec_stops = [
            {
                "pos": float(stop["pos"]),
                "L": float(stop["L"]),
                "C": float(stop["C"]),
                "H": float(stop["H"]),
            }
            for stop in stops
        ]
        interp_mix = 1.0 if self.palette_editor_interp_mix >= 0.5 else 0.0
        return {
            "space": "oklch",
            "relative_chroma": bool(self.palette_editor_relative_chroma),
            "interp_mix": float(interp_mix),
            "stops": spec_stops,
        }

    def _palette_editor_commit_palette(self, persist: bool = False) -> bool:
        if self.palette_editor_palette_name is None:
            return False

        runtime_dirty = self.palette_editor_dirty
        needs_persist = persist and self.palette_editor_persist_dirty
        if not runtime_dirty and not needs_persist:
            return False

        spec = self._palette_editor_build_spec()
        with self.app.state_lock:
            set_palette_spec(self.palette_editor_palette_name, spec, persist=persist)

        if runtime_dirty:
            updated = self.app.display_pipeline.texture_manager.update_palette_colormap(self.palette_editor_palette_name)
            if not updated:
                self.palette_editor_needs_colormap_rebuild = True

        self.palette_editor_dirty = False
        if persist:
            self.palette_editor_persist_dirty = False
        return runtime_dirty

    # ------------------------------------------------------------------
    # Refresh and debounce
    # ------------------------------------------------------------------

    def _request_palette_editor_refresh(self, force: bool = False) -> None:
        if not self.palette_editor_active:
            return

        self.palette_editor_dirty = True
        self.palette_editor_persist_dirty = True
        now = time.time()
        if force or (now - self.palette_editor_last_refresh_time) >= self.palette_editor_refresh_throttle:
            self._apply_palette_editor_refresh()
        else:
            self.palette_editor_refresh_pending = True

    def _apply_palette_editor_refresh(self, persist: bool = False) -> None:
        if not self.palette_editor_active:
            self.palette_editor_refresh_pending = False
            return

        had_persist_changes = persist and (self.palette_editor_dirty or self.palette_editor_persist_dirty)
        changed = self._palette_editor_commit_palette(persist=persist)
        if had_persist_changes:
            self.palette_editor_needs_colormap_rebuild = True
        if changed:
            self.app.state_manager.update(StateKey.PALETTE, self.palette_editor_palette_name)
            self.palette_editor_last_refresh_time = time.time()
        self.palette_editor_refresh_pending = False

    def check_palette_editor_debounce(self) -> None:
        """Check if a pending palette editor refresh should be applied.

        Call this from the main loop to handle debounced updates.
        """
        if not self.palette_editor_refresh_pending:
            return
        now = time.time()
        if (now - self.palette_editor_last_refresh_time) >= self.palette_editor_refresh_throttle:
            self._apply_palette_editor_refresh()

    def _finalize_palette_editor_colormaps(self) -> None:
        if not self.palette_editor_needs_colormap_rebuild:
            return
        self.app.display_pipeline.texture_manager.rebuild_colormaps()
        self._rebuild_palette_popup()
        self._update_palette_preview_buttons()
        self.palette_editor_needs_colormap_rebuild = False

    # ------------------------------------------------------------------
    # Texture generation
    # ------------------------------------------------------------------

    def _palette_editor_generate_gradient_texture(self) -> np.ndarray:
        bar_width = max(1, int(self.palette_editor_width - 2 * self.palette_editor_gradient_padding))
        bar_height = max(1, int(self.palette_editor_gradient_bar_bottom - self.palette_editor_gradient_bar_top))

        lut = build_oklch_lut(
            self.palette_editor_stops,
            size=bar_width,
            relative_chroma=self.palette_editor_relative_chroma,
            interp_mix=self.palette_editor_interp_mix,
        )
        rgba = np.ones((bar_height, bar_width, 4), dtype=np.float32)
        rgba[:, :, :3] = lut[np.newaxis, :, :]
        return rgba

    def _palette_editor_generate_slice_texture(self) -> np.ndarray:
        stop = self._palette_editor_get_selected_stop()
        L_val = float(stop["L"]) if stop else 0.5
        L_arr = np.full_like(self.palette_editor_c_mesh, L_val, dtype=np.float32)

        if self.palette_editor_relative_chroma:
            C_abs = self._palette_editor_chroma_rel_to_abs_array(
                L_arr,
                self.palette_editor_h_mesh,
                self.palette_editor_c_mesh,
            )
        else:
            C_abs = np.clip(self.palette_editor_c_mesh, 0.0, None)

        rgb = gamut_map_to_srgb(L_arr, C_abs, self.palette_editor_h_mesh, method="compress")
        rgba = np.ones((self.palette_editor_slice_height, self.palette_editor_width, 4), dtype=np.float32)
        rgba[..., :3] = np.clip(rgb, 0.0, 1.0)
        return rgba

    def _palette_editor_generate_l_gradient_texture(self) -> np.ndarray:
        width = max(1, int(self.palette_editor_lbar_width))
        height = max(1, int(self.palette_editor_lbar_height))

        stop = self._palette_editor_get_selected_stop()
        C_val = float(stop["C"]) if stop else 0.0
        H_val = float(stop["H"]) if stop else 0.0

        L_arr = np.linspace(0.0, 1.0, width, dtype=np.float32)
        H_arr = np.full(width, H_val, dtype=np.float32)

        if self.palette_editor_relative_chroma:
            C_rel = float(np.clip(C_val, 0.0, 1.0))
            max_c = max_chroma_fast(L_arr, H_arr)
            C_abs = max_c * C_rel
        else:
            C_abs = np.full(width, float(np.clip(C_val, 0.0, self.palette_editor_c_max_absolute)), dtype=np.float32)

        rgb = gamut_map_to_srgb(L_arr, C_abs, H_arr, method="compress")
        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = np.clip(rgb, 0.0, 1.0)[np.newaxis, :, :]
        return rgba

    # ------------------------------------------------------------------
    # Texture updates
    # ------------------------------------------------------------------

    def _palette_editor_update_gradient_texture(self) -> None:
        if dpg is None:
            return
        texture_registry = self.app.display_pipeline.texture_manager.texture_registry_id
        if texture_registry is None:
            return

        rgba = self._palette_editor_generate_gradient_texture()
        data = rgba.ravel()
        width = rgba.shape[1]
        height = rgba.shape[0]
        if self.palette_editor_gradient_texture_id is None or not dpg.does_item_exist(self.palette_editor_gradient_texture_id):
            self.palette_editor_gradient_texture_id = dpg.add_dynamic_texture(
                width,
                height,
                data,
                parent=texture_registry,
            )
        else:
            dpg.set_value(self.palette_editor_gradient_texture_id, data)

    def _palette_editor_update_slice_texture(self) -> None:
        if dpg is None:
            return
        texture_registry = self.app.display_pipeline.texture_manager.texture_registry_id
        if texture_registry is None:
            return

        rgba = self._palette_editor_generate_slice_texture()
        data = rgba.ravel()
        width = rgba.shape[1]
        height = rgba.shape[0]
        if self.palette_editor_slice_texture_id is None or not dpg.does_item_exist(self.palette_editor_slice_texture_id):
            self.palette_editor_slice_texture_id = dpg.add_dynamic_texture(
                width,
                height,
                data,
                parent=texture_registry,
            )
        else:
            dpg.set_value(self.palette_editor_slice_texture_id, data)

    def _palette_editor_update_l_gradient_texture(self) -> None:
        if dpg is None:
            return
        texture_registry = self.app.display_pipeline.texture_manager.texture_registry_id
        if texture_registry is None:
            return

        rgba = self._palette_editor_generate_l_gradient_texture()
        data = rgba.ravel()
        width = rgba.shape[1]
        height = rgba.shape[0]
        if self.palette_editor_l_gradient_texture_id is None or not dpg.does_item_exist(self.palette_editor_l_gradient_texture_id):
            self.palette_editor_l_gradient_texture_id = dpg.add_dynamic_texture(
                width,
                height,
                data,
                parent=texture_registry,
            )
        else:
            dpg.set_value(self.palette_editor_l_gradient_texture_id, data)

    # ------------------------------------------------------------------
    # Drawlist updates
    # ------------------------------------------------------------------

    def _palette_editor_update_gradient_drawlist(self) -> None:
        if dpg is None or self.palette_editor_gradient_drawlist_id is None:
            return
        if not dpg.does_item_exist(self.palette_editor_gradient_drawlist_id):
            return

        dpg.delete_item(self.palette_editor_gradient_drawlist_id, children_only=True)

        bar_left = self.palette_editor_gradient_padding
        bar_right = self.palette_editor_width - self.palette_editor_gradient_padding
        bar_top = self.palette_editor_gradient_bar_top
        bar_bottom = self.palette_editor_gradient_bar_bottom

        if self.palette_editor_gradient_texture_id is not None:
            dpg.draw_image(
                self.palette_editor_gradient_texture_id,
                (bar_left, bar_top),
                (bar_right, bar_bottom),
                parent=self.palette_editor_gradient_drawlist_id,
            )

        dpg.draw_rectangle(
            (bar_left, bar_top),
            (bar_right, bar_bottom),
            color=(100, 100, 100, 255),
            thickness=1,
            parent=self.palette_editor_gradient_drawlist_id,
        )

        for is_selected_pass in (False, True):
            for stop in self.palette_editor_stops:
                is_selected = stop["id"] == self.palette_editor_selected_stop_id
                if is_selected != is_selected_pass:
                    continue

                x = self._palette_editor_stop_to_x(stop["pos"])
                y = self.palette_editor_handle_center_y
                r, g, b = self._palette_editor_oklch_to_rgb255(stop["L"], stop["C"], stop["H"])
                if is_selected:
                    dpg.draw_circle(
                        (x, y),
                        self.palette_editor_handle_radius + 3,
                        color=(200, 200, 200, 220),
                        thickness=2,
                        parent=self.palette_editor_gradient_drawlist_id,
                    )
                dpg.draw_circle(
                    (x, y),
                    self.palette_editor_handle_radius,
                    color=(30, 30, 30, 255),
                    fill=(r, g, b, 255),
                    thickness=1,
                    parent=self.palette_editor_gradient_drawlist_id,
                )

    def _palette_editor_update_slice_drawlist(self) -> None:
        if dpg is None or self.palette_editor_slice_drawlist_id is None:
            return
        if not dpg.does_item_exist(self.palette_editor_slice_drawlist_id):
            return

        dpg.delete_item(self.palette_editor_slice_drawlist_id, children_only=True)
        if self.palette_editor_slice_texture_id is not None:
            dpg.draw_image(
                self.palette_editor_slice_texture_id,
                (0, 0),
                (self.palette_editor_width, self.palette_editor_slice_height),
                parent=self.palette_editor_slice_drawlist_id,
            )

        stop = self._palette_editor_get_selected_stop()
        if stop is None:
            return

        h_normalized = float(stop["H"]) % 360.0
        x = float(np.clip((h_normalized / 360.0) * self.palette_editor_width, 0.0, self.palette_editor_width - 1))
        y = float(np.clip((stop["C"] / self.palette_editor_c_max_display) * self.palette_editor_slice_height, 0.0, self.palette_editor_slice_height))
        color = (240, 240, 240, 220)
        dpg.draw_line(
            (0, y),
            (self.palette_editor_width, y),
            color=color,
            thickness=1,
            parent=self.palette_editor_slice_drawlist_id,
        )
        dpg.draw_line(
            (x, 0),
            (x, self.palette_editor_slice_height),
            color=color,
            thickness=1,
            parent=self.palette_editor_slice_drawlist_id,
        )
        dpg.draw_circle(
            (x, y),
            4,
            color=color,
            thickness=1,
            parent=self.palette_editor_slice_drawlist_id,
        )

    def _palette_editor_update_l_gradient_drawlist(self) -> None:
        if dpg is None or self.palette_editor_l_gradient_drawlist_id is None:
            return
        if not dpg.does_item_exist(self.palette_editor_l_gradient_drawlist_id):
            return

        dpg.delete_item(self.palette_editor_l_gradient_drawlist_id, children_only=True)
        if self.palette_editor_l_gradient_texture_id is not None:
            dpg.draw_image(
                self.palette_editor_l_gradient_texture_id,
                (0, 0),
                (self.palette_editor_lbar_width, self.palette_editor_lbar_height),
                parent=self.palette_editor_l_gradient_drawlist_id,
            )

        dpg.draw_rectangle(
            (0, 0),
            (self.palette_editor_lbar_width, self.palette_editor_lbar_height),
            color=(90, 90, 90, 200),
            thickness=1,
            parent=self.palette_editor_l_gradient_drawlist_id,
        )

    # ------------------------------------------------------------------
    # Control sync and visual refresh
    # ------------------------------------------------------------------

    def _palette_editor_sync_controls(self, skip_slider_sync: bool = False) -> None:
        if dpg is None:
            return

        stop = self._palette_editor_get_selected_stop()
        if stop is None:
            self._set_palette_editor_controls_enabled(False)
            return

        self._set_palette_editor_controls_enabled(True)
        if self._palette_editor_any_slider_active():
            skip_slider_sync = True

        if not skip_slider_sync:
            self.palette_editor_syncing = True
            try:
                if (
                    self.palette_editor_l_slider_id is not None
                    and dpg.does_item_exist(self.palette_editor_l_slider_id)
                    and not self._palette_editor_item_active(self.palette_editor_l_slider_id)
                ):
                    dpg.set_value(self.palette_editor_l_slider_id, float(stop["L"]))
                if (
                    self.palette_editor_c_slider_id is not None
                    and dpg.does_item_exist(self.palette_editor_c_slider_id)
                    and not self._palette_editor_item_active(self.palette_editor_c_slider_id)
                ):
                    dpg.set_value(self.palette_editor_c_slider_id, float(stop["C"]))
                if (
                    self.palette_editor_h_slider_id is not None
                    and dpg.does_item_exist(self.palette_editor_h_slider_id)
                    and not self._palette_editor_item_active(self.palette_editor_h_slider_id)
                ):
                    dpg.set_value(self.palette_editor_h_slider_id, float(stop["H"]))
            finally:
                self.palette_editor_syncing = False

    def _palette_editor_refresh_visuals(self, skip_slider_sync: bool = False) -> None:
        self._palette_editor_update_gradient_texture()
        self._palette_editor_update_slice_texture()
        self._palette_editor_update_l_gradient_texture()
        self._palette_editor_update_gradient_drawlist()
        self._palette_editor_update_slice_drawlist()
        self._palette_editor_update_l_gradient_drawlist()
        self._palette_editor_sync_controls(skip_slider_sync=skip_slider_sync)

    # ------------------------------------------------------------------
    # Mouse handlers
    # ------------------------------------------------------------------

    def _on_palette_editor_mouse_down(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)

        coords = self._palette_editor_drawlist_local_coords(
            self.palette_editor_gradient_drawlist_id,
            mouse_x,
            mouse_y,
            clamp=False,
        )
        if coords is not None:
            local_x, local_y = coords
            handle_idx = self._palette_editor_hit_test_handle(local_x, local_y)
            if handle_idx is not None:
                self.palette_editor_selected_stop_id = self.palette_editor_stops[handle_idx]["id"]
                self.palette_editor_dragging_stop_id = self.palette_editor_selected_stop_id
                self.palette_editor_is_dragging = True
                self._palette_editor_update_gradient_drawlist()
                self._palette_editor_sync_controls()
                return

        coords = self._palette_editor_drawlist_local_coords(
            self.palette_editor_slice_drawlist_id,
            mouse_x,
            mouse_y,
            clamp=False,
        )
        if coords is not None:
            local_x, local_y = coords
            stop = self._palette_editor_get_selected_stop()
            if stop is None:
                return
            self.palette_editor_slice_drag_active = True
            new_h = float(np.clip((local_x / self.palette_editor_width) * 360.0, 0.0, 360.0))
            new_c = float(np.clip((local_y / self.palette_editor_slice_height) * self.palette_editor_c_max_display, 0.0, self.palette_editor_c_max_display))
            stop["H"] = new_h
            stop["C"] = new_c
            self._palette_editor_refresh_visuals()
            self._request_palette_editor_refresh()

    def _on_palette_editor_mouse_drag(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)

        if self.palette_editor_is_dragging and self.palette_editor_dragging_stop_id is not None:
            coords = self._palette_editor_drawlist_local_coords(
                self.palette_editor_gradient_drawlist_id,
                mouse_x,
                mouse_y,
                clamp=True,
            )
            if coords is None:
                return
            local_x, _ = coords
            idx = self._palette_editor_get_stop_index(self.palette_editor_dragging_stop_id)
            if idx is None:
                return
            new_pos = self._palette_editor_x_to_pos(local_x)
            min_pos = 0.0
            max_pos = 1.0
            if idx > 0:
                min_pos = self.palette_editor_stops[idx - 1]["pos"] + 0.005
            if idx < len(self.palette_editor_stops) - 1:
                max_pos = self.palette_editor_stops[idx + 1]["pos"] - 0.005
            self.palette_editor_stops[idx]["pos"] = float(np.clip(new_pos, min_pos, max_pos))
            self._palette_editor_sort_stops()
            self._palette_editor_update_gradient_texture()
            self._palette_editor_update_gradient_drawlist()
            self._palette_editor_sync_controls()
            self._request_palette_editor_refresh()
            return

        if self.palette_editor_slice_drag_active:
            coords = self._palette_editor_drawlist_local_coords(
                self.palette_editor_slice_drawlist_id,
                mouse_x,
                mouse_y,
                clamp=True,
            )
            if coords is None:
                return
            local_x, local_y = coords
            stop = self._palette_editor_get_selected_stop()
            if stop is None:
                return
            stop["H"] = float(np.clip((local_x / self.palette_editor_width) * 360.0, 0.0, 360.0))
            stop["C"] = float(np.clip((local_y / self.palette_editor_slice_height) * self.palette_editor_c_max_display, 0.0, self.palette_editor_c_max_display))
            self._palette_editor_refresh_visuals()
            self._request_palette_editor_refresh()

    def _on_palette_editor_mouse_release(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return

        if self.palette_editor_is_dragging or self.palette_editor_slice_drag_active:
            self.palette_editor_is_dragging = False
            self.palette_editor_dragging_stop_id = None
            self.palette_editor_slice_drag_active = False
            self._palette_editor_sort_stops()
            self._palette_editor_refresh_visuals()
            if self.palette_editor_refresh_pending or self.palette_editor_dirty:
                self._apply_palette_editor_refresh()

    def _on_palette_editor_mouse_double_click(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        coords = self._palette_editor_drawlist_local_coords(
            self.palette_editor_gradient_drawlist_id,
            mouse_x,
            mouse_y,
            clamp=False,
        )
        if coords is None:
            return
        local_x, local_y = coords

        handle_idx = self._palette_editor_hit_test_handle(local_x, local_y)
        if handle_idx is not None:
            stop_id = self.palette_editor_stops[handle_idx]["id"]
            self._palette_editor_remove_stop(stop_id)
            self._palette_editor_refresh_visuals()
            self._request_palette_editor_refresh(force=True)
            return

        if self.palette_editor_gradient_bar_top <= local_y <= self.palette_editor_gradient_bar_bottom + self.palette_editor_handle_radius:
            pos = self._palette_editor_x_to_pos(local_x)
            self._palette_editor_add_stop(pos)
            self._palette_editor_refresh_visuals()
            self._request_palette_editor_refresh(force=True)

    # ------------------------------------------------------------------
    # Slider callbacks
    # ------------------------------------------------------------------

    def on_palette_editor_l_slider(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return
        if self.palette_editor_syncing:
            return
        stop = self._palette_editor_get_selected_stop()
        if stop is None:
            return
        stop["L"] = float(np.clip(float(app_data), 0.0, 1.0))
        self._palette_editor_refresh_visuals(skip_slider_sync=True)
        self._request_palette_editor_refresh()

    def on_palette_editor_c_slider(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return
        if self.palette_editor_syncing:
            return
        stop = self._palette_editor_get_selected_stop()
        if stop is None:
            return
        max_value = 1.0 if self.palette_editor_relative_chroma else self.palette_editor_c_max_absolute
        stop["C"] = float(np.clip(float(app_data), 0.0, max_value))
        self._palette_editor_refresh_visuals(skip_slider_sync=True)
        self._request_palette_editor_refresh()

    def on_palette_editor_h_slider(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return
        if self.palette_editor_syncing:
            return
        stop = self._palette_editor_get_selected_stop()
        if stop is None:
            return
        stop["H"] = float(np.mod(float(app_data), 360.0))
        self._palette_editor_refresh_visuals(skip_slider_sync=True)
        self._request_palette_editor_refresh()

    def on_palette_editor_interp_mode(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active or self.palette_editor_interp_mode_id is None:
            return
        mode = dpg.get_value(self.palette_editor_interp_mode_id)
        if mode == "Absolute":
            self.palette_editor_interp_mix = 0.0
        else:
            self.palette_editor_interp_mix = 1.0
        self._palette_editor_refresh_visuals()
        self._request_palette_editor_refresh(force=True)

    # ------------------------------------------------------------------
    # Preview state and control
    # ------------------------------------------------------------------

    def _get_palette_preview_state(self, for_region: bool) -> tuple[str, Optional[str]]:
        """Return label + colormap tag for the palette preview button."""
        texture_manager = self.app.display_pipeline.texture_manager
        grayscale_tag = texture_manager.grayscale_colormap_tag

        with self.app.state_lock:
            if for_region:
                region_style = self._get_current_region_style_unlocked()
                if region_style and region_style.enabled and region_style.use_palette:
                    palette_name = region_style.palette
                    return palette_name, texture_manager.palette_colormaps.get(palette_name) or grayscale_tag

                if self.app.state.display_settings.color_enabled:
                    palette_name = self.app.state.display_settings.palette
                    return palette_name, texture_manager.palette_colormaps.get(palette_name) or grayscale_tag

                return "Grayscale", grayscale_tag

            if self.app.state.display_settings.color_enabled:
                palette_name = self.app.state.display_settings.palette
                return palette_name, texture_manager.palette_colormaps.get(palette_name) or grayscale_tag

            return "Grayscale", grayscale_tag

    def _update_palette_preview_buttons(self) -> None:
        """Refresh palette preview labels and colormap bindings."""
        if dpg is None:
            return

        if self.global_palette_preview_button_id is not None:
            label, tag = self._get_palette_preview_state(for_region=False)
            if dpg.does_item_exist(self.global_palette_preview_button_id):
                dpg.configure_item(self.global_palette_preview_button_id, label=label)
                if tag is not None:
                    dpg.bind_colormap(self.global_palette_preview_button_id, tag)

        if self.region_palette_preview_button_id is not None:
            label, tag = self._get_palette_preview_state(for_region=True)
            if dpg.does_item_exist(self.region_palette_preview_button_id):
                dpg.configure_item(self.region_palette_preview_button_id, label=label)
                if tag is not None:
                    dpg.bind_colormap(self.region_palette_preview_button_id, tag)

        self._update_palette_popup_controls()

    def _get_editable_palette_name(self, for_region: bool) -> Optional[str]:
        """Return palette name if edit/duplicate is valid in this context."""
        with self.app.state_lock:
            if for_region:
                region_style = self._get_current_region_style_unlocked()
                if region_style and region_style.enabled and region_style.use_palette:
                    return region_style.palette
                return None

            if self.app.state.display_settings.color_enabled:
                return self.app.state.display_settings.palette
            return None

    def _update_palette_popup_controls(self) -> None:
        """Enable/disable Edit/Duplicate based on current context."""
        if dpg is None:
            return

        global_editable = self._get_editable_palette_name(for_region=False) is not None
        region_editable = self._get_editable_palette_name(for_region=True) is not None
        if self.palette_editor_active:
            global_editable = False
            region_editable = False

        if dpg.does_item_exist("global_palette_edit_btn"):
            dpg.configure_item("global_palette_edit_btn", enabled=global_editable)
        if dpg.does_item_exist("global_palette_duplicate_btn"):
            dpg.configure_item("global_palette_duplicate_btn", enabled=global_editable)
        if dpg.does_item_exist("region_palette_edit_btn"):
            dpg.configure_item("region_palette_edit_btn", enabled=region_editable)
        if dpg.does_item_exist("region_palette_duplicate_btn"):
            dpg.configure_item("region_palette_duplicate_btn", enabled=region_editable)

    def _set_palette_editor_state(self, active: bool, palette_name: Optional[str] = None, for_region: bool = False) -> None:
        """Show/hide the palette editor and sync controls."""
        if dpg is None:
            return

        self.palette_editor_active = active
        self.palette_editor_palette_name = palette_name if active else None
        self.palette_editor_for_region = for_region if active else False

        if self.palette_editor_group_id is not None and dpg.does_item_exist(self.palette_editor_group_id):
            dpg.configure_item(self.palette_editor_group_id, show=active)

        if self.palette_editor_title_id is not None and dpg.does_item_exist(self.palette_editor_title_id):
            title = f"Editing: {palette_name}" if active and palette_name else "Editing: (none)"
            dpg.set_value(self.palette_editor_title_id, title)

        if self.global_palette_preview_button_id is not None and dpg.does_item_exist(self.global_palette_preview_button_id):
            dpg.configure_item(self.global_palette_preview_button_id, enabled=not active)
        if self.region_palette_preview_button_id is not None and dpg.does_item_exist(self.region_palette_preview_button_id):
            dpg.configure_item(self.region_palette_preview_button_id, enabled=not active)

        if not active:
            self._set_palette_editor_controls_enabled(False)

        self._update_palette_popup_controls()

    # ------------------------------------------------------------------
    # Palette action callbacks
    # ------------------------------------------------------------------

    def on_palette_edit(self, sender=None, app_data=None, user_data=None) -> None:
        """Open the inline palette editor for the active palette."""
        if dpg is None:
            return

        for_region = bool(user_data)
        palette_name = self._get_editable_palette_name(for_region)
        if palette_name is None:
            return

        self._set_palette_editor_state(True, palette_name, for_region)
        self._palette_editor_load_palette(palette_name)
        popup_tag = "region_palette_popup" if for_region else "global_palette_popup"
        if dpg.does_item_exist(popup_tag):
            dpg.configure_item(popup_tag, show=False)

    def on_palette_editor_done(self, sender=None, app_data=None) -> None:
        """Exit palette edit mode."""
        if dpg is None:
            return
        if self.palette_editor_dirty or self.palette_editor_persist_dirty:
            self._apply_palette_editor_refresh(persist=True)
        self._finalize_palette_editor_colormaps()
        self._set_palette_editor_state(False)
        self.palette_editor_refresh_pending = False

    def _generate_duplicate_palette_name(self, base: str) -> str:
        existing = set(list_color_palettes())
        candidate = f"{base} Copy"
        if candidate not in existing:
            return candidate
        index = 2
        while True:
            candidate = f"{base} Copy {index}"
            if candidate not in existing:
                return candidate
            index += 1

    def on_palette_duplicate(self, sender=None, app_data=None, user_data=None) -> None:
        """Duplicate the active palette and enter edit mode on the copy."""
        if dpg is None:
            return

        for_region = bool(user_data)
        if self.palette_editor_active:
            return

        palette_name = self._get_editable_palette_name(for_region)
        if palette_name is None:
            return

        spec = get_palette_spec(palette_name)
        if spec is None:
            return

        new_name = self._generate_duplicate_palette_name(palette_name)
        with self.app.state_lock:
            set_palette_spec(new_name, spec)
            self.app.display_pipeline.texture_manager.rebuild_colormaps()
        self._rebuild_palette_popup()

        if for_region:
            with self.app.state_lock:
                selected = self.app.state.get_selected()
                region_type = self.app.state.selected_region_type
                boundary_id = selected.id if selected else None
                global_b = self.app.state.display_settings.brightness
                global_c = self.app.state.display_settings.contrast
                global_g = self.app.state.display_settings.gamma
            if boundary_id is not None:
                self.app.state_manager.update(StateKey.REGION_STYLE, {
                    "enabled": True,
                    "use_palette": True,
                    "palette": new_name,
                    "brightness": global_b,
                    "contrast": global_c,
                    "gamma": global_g,
                }, context=(boundary_id, region_type))
        else:
            self.app.state_manager.update(StateKey.COLOR_ENABLED, True)
            self.app.state_manager.update(StateKey.PALETTE, new_name)

        self.update_context_ui()
        self._update_palette_preview_buttons()

        popup_tag = "region_palette_popup" if for_region else "global_palette_popup"
        if dpg.does_item_exist(popup_tag):
            dpg.configure_item(popup_tag, show=False)

        self._set_palette_editor_state(True, new_name, for_region)
        self._palette_editor_load_palette(new_name)

    def on_global_grayscale(self, sender=None, app_data=None) -> None:
        """Handle global 'Grayscale (No Color)' button."""
        if dpg is None:
            return
        if self.palette_editor_active:
            return

        self.app.state_manager.update(StateKey.COLOR_ENABLED, False)

        self._update_palette_preview_buttons()
        dpg.configure_item("global_palette_popup", show=False)

    def on_global_palette_button(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle global colormap button click."""
        if dpg is None or user_data is None:
            return
        if self.palette_editor_active:
            return

        palette_name = user_data
        self.app.state_manager.update(StateKey.COLOR_ENABLED, True)
        self.app.state_manager.update(StateKey.PALETTE, palette_name)

        self._update_palette_preview_buttons()
        dpg.configure_item("global_palette_popup", show=False)

    def on_region_use_global(self, sender=None, app_data=None) -> None:
        """Handle region 'Use Global' button - disables override."""
        if dpg is None:
            return
        if self.palette_editor_active:
            return

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            region_type = self.app.state.selected_region_type
            boundary_id = selected.id if selected else None

        if boundary_id is not None:
            self.app.state_manager.update(StateKey.REGION_STYLE, {
                "enabled": False,
                "brightness": None,
                "contrast": None,
                "gamma": None,
                "lightness_expr": None,
            }, context=(boundary_id, region_type))

        dpg.configure_item("region_palette_popup", show=False)

        self.update_context_ui()  # Update slider states

    def on_region_palette_button(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle region colormap button click - also enables override."""
        if dpg is None or user_data is None:
            return
        if self.palette_editor_active:
            return

        palette_name = user_data
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            region_type = self.app.state.selected_region_type
            boundary_id = selected.id if selected else None
            global_b = self.app.state.display_settings.brightness
            global_c = self.app.state.display_settings.contrast
            global_g = self.app.state.display_settings.gamma

        if boundary_id is not None:
            self.app.state_manager.update(StateKey.REGION_STYLE, {
                "enabled": True,
                "use_palette": True,
                "palette": palette_name,
                "brightness": global_b,
                "contrast": global_c,
                "gamma": global_g,
            }, context=(boundary_id, region_type))

        dpg.configure_item("region_palette_popup", show=False)

        self.update_context_ui()  # Update slider states

    # ------------------------------------------------------------------
    # Delete confirmation modal
    # ------------------------------------------------------------------

    def _ensure_delete_confirmation_modal(self) -> None:
        """Create the delete confirmation modal if it doesn't exist."""
        if dpg is None or self.delete_confirmation_modal_id is not None:
            return

        with dpg.window(
            label="Delete Palette?",
            modal=True,
            show=False,
            tag="delete_palette_modal",
            no_resize=True,
            no_move=False,
            no_close=True,
            no_collapse=True,
            width=350,
            height=150,
            no_open_over_existing_popup=False,
        ) as modal:
            self.delete_confirmation_modal_id = modal

            dpg.add_text("", tag="delete_palette_modal_text")
            dpg.add_text("This cannot be undone.", color=(200, 200, 0))
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Delete",
                    width=75,
                    callback=self._confirm_delete_palette
                )
                dpg.add_button(
                    label="Cancel",
                    width=75,
                    callback=self._cancel_delete_palette
                )

    def _open_delete_confirmation(self, palette_name: str) -> None:
        """Open the delete confirmation modal for a specific palette."""
        if dpg is None:
            return

        self._ensure_delete_confirmation_modal()
        self.pending_delete_palette = palette_name

        dpg.set_value("delete_palette_modal_text", f"Delete '{palette_name}'?")

        # Center the modal on screen
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        modal_width = 350
        modal_height = 150
        pos_x = (viewport_width - modal_width) // 2
        pos_y = (viewport_height - modal_height) // 2

        dpg.configure_item(self.delete_confirmation_modal_id, pos=[pos_x, pos_y])
        dpg.configure_item(self.delete_confirmation_modal_id, show=True)

    def _cancel_delete_palette(self, sender=None, app_data=None) -> None:
        """Cancel palette deletion."""
        if dpg is None or self.delete_confirmation_modal_id is None:
            return
        dpg.configure_item(self.delete_confirmation_modal_id, show=False)
        self.pending_delete_palette = None

    def _confirm_delete_palette(self, sender=None, app_data=None, user_data=None) -> None:
        """Actually delete the palette after confirmation."""
        if dpg is None or self.pending_delete_palette is None:
            return

        palette_name = self.pending_delete_palette
        delete_palette(palette_name)

        # Rebuild DPG colormaps to reflect deletion
        self.app.display_pipeline.texture_manager.rebuild_colormaps()

        # Rebuild the palette popup menu
        self._rebuild_palette_popup()
        self._update_palette_preview_buttons()

        # Close the confirmation modal
        dpg.configure_item(self.delete_confirmation_modal_id, show=False)
        self.pending_delete_palette = None

    def _rebuild_palette_popup(self) -> None:
        """Rebuild palette popup after deletion."""
        if dpg is None:
            return

        # Delete old scrolling window content
        if dpg.does_item_exist("global_palette_scrolling_window"):
            dpg.delete_item("global_palette_scrolling_window", children_only=True)

        # Rebuild colormap buttons
        palette_names = list(list_color_palettes())

        for palette_name in palette_names:
            colormap_tag = self.app.display_pipeline.texture_manager.palette_colormaps.get(palette_name)
            if not colormap_tag:
                continue

            # Create a group for each palette entry (colormap button + delete button)
            with dpg.group(horizontal=True, parent="global_palette_scrolling_window"):
                btn = dpg.add_colormap_button(
                    label=palette_name,
                    width=310,
                    height=25,
                    callback=self.on_global_palette_button,
                    user_data=palette_name,
                    tag=f"global_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                )
                dpg.bind_colormap(btn, colormap_tag)

                # Add delete button that opens separate modal
                dpg.add_button(
                    label="X",
                    width=30,
                    height=25,
                    callback=lambda s, a, u: self._open_delete_confirmation(u),
                    user_data=palette_name
                )

        # Rebuild region palette list too
        if dpg.does_item_exist("region_palette_scrolling_window"):
            dpg.delete_item("region_palette_scrolling_window", children_only=True)

            for palette_name in palette_names:
                colormap_tag = self.app.display_pipeline.texture_manager.palette_colormaps.get(palette_name)
                if not colormap_tag:
                    continue
                btn = dpg.add_colormap_button(
                    label=palette_name,
                    width=350,
                    height=25,
                    callback=self.on_region_palette_button,
                    user_data=palette_name,
                    tag=f"region_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                    parent="region_palette_scrolling_window",
                )
                dpg.bind_colormap(btn, colormap_tag)
