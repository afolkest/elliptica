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
import json

import numpy as np
import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flowcol.serialization import load_project, load_render_cache
from flowcol.render import _RUNTIME_PALETTES, _get_palette_lut, add_palette, USER_PALETTES_PATH
from flowcol.gpu import GPUContext
from flowcol.gpu.pipeline import build_base_rgb_gpu

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

        # Custom gradient state
        self.gradient_stops: list[dict] = []
        self._next_stop_id: int = 0
        self.selected_stop_id: Optional[int] = None
        self.dragging_stop_id: Optional[int] = None
        self.use_custom_gradient: bool = False
        self.custom_gradient_lut: Optional[np.ndarray] = None

        # Gradient UI resources
        self.gradient_texture_id: Optional[int] = None
        self.gradient_drawlist_id: Optional[int] = None
        self.gradient_handler_id: Optional[int] = None
        self.gradient_color_picker_id: Optional[int] = None
        self.gradient_position_slider_id: Optional[int] = None
        self.use_gradient_checkbox_id: Optional[int] = None
        self.gradient_name_input_id: Optional[int] = None
        self.gradient_selected_text_id: Optional[int] = None
        self.gradient_default_name: str = "Custom Gradient"

        # Gradient layout parameters
        self.gradient_width = 320
        self.gradient_height = 140
        self.gradient_bar_padding = 20
        self.gradient_bar_top = 40
        self.gradient_bar_bottom = 80
        self.gradient_handle_radius = 9
        self._handle_positions: dict[int, tuple[float, float]] = {}

        # Gradient persistence & limits
        self.gradient_metadata_path = USER_PALETTES_PATH.with_name("palettes_gradients.json")
        self.gradient_max_initial_stops = 12
        self.gradient_min_spacing = 1e-3
        self.gradient_metadata: dict[str, list[dict]] = self._load_gradient_metadata()

        # Initialize gradient stops from current palette
        self._init_gradient_from_palette(self.current_palette)

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
        # Use custom gradient if enabled, otherwise fall back to library palette
        if self.use_custom_gradient and self.custom_gradient_lut is not None:
            lut_numpy = self.custom_gradient_lut
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

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------

    def _load_gradient_metadata(self) -> dict[str, list[dict]]:
        """Load saved gradient stop metadata from disk."""
        path = getattr(self, "gradient_metadata_path", None)
        if path is None or not path.exists():
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:  # pragma: no cover - logged for observability
            print(f"⚠ Failed to read gradient metadata: {exc}")
            return {}

        metadata: dict[str, list[dict]] = {}
        for name, stops in raw.items():
            if not isinstance(stops, list):
                continue
            sanitized = self._sanitize_stops(stops)
            if sanitized:
                metadata[name] = sanitized
        return metadata

    def _persist_gradient_metadata(self) -> None:
        """Write gradient stop metadata back to disk."""
        path = self.gradient_metadata_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                name: [
                    {"pos": float(stop["pos"]), "color": [float(c) for c in stop["color"]]}
                    for stop in stops
                ]
                for name, stops in self.gradient_metadata.items()
                if stops
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as exc:  # pragma: no cover - logged for observability
            print(f"✗ Failed to write gradient metadata: {exc}")

    def _record_gradient_metadata(self, name: str) -> None:
        """Cache current stops for a palette name and persist to disk."""
        stops = [
            {
                "pos": float(stop["pos"]),
                "color": [float(stop["color"][0]), float(stop["color"][1]), float(stop["color"][2])],
            }
            for stop in self.gradient_stops
        ]
        self.gradient_metadata[name] = self._sanitize_stops(stops)
        self._persist_gradient_metadata()

    def _sanitize_stops(self, stops_input: list) -> list[dict]:
        """Normalize list of stops to sorted structure with clamped values."""
        cleaned: list[dict] = []
        for stop in stops_input or []:
            if isinstance(stop, dict):
                pos = stop.get("pos")
                color = stop.get("color")
            elif isinstance(stop, (list, tuple)) and len(stop) >= 4:
                pos = stop[0]
                color = stop[1:4]
            else:
                continue

            if pos is None or color is None:
                continue

            try:
                pos_f = float(pos)
            except (TypeError, ValueError):
                continue

            try:
                color_vals = np.array(color[:3], dtype=np.float32)
            except (TypeError, ValueError):
                continue

            if np.isnan(color_vals).any():
                continue

            if float(np.max(color_vals)) > 1.0 + 1e-3:
                color_vals = color_vals / 255.0

            color_vals = np.clip(color_vals, 0.0, 1.0)
            cleaned.append({"pos": pos_f, "color": color_vals.tolist()})

        if not cleaned:
            return [
                {"pos": 0.0, "color": [0.0, 0.0, 0.0]},
                {"pos": 1.0, "color": [1.0, 1.0, 1.0]},
            ]

        cleaned.sort(key=lambda s: s["pos"])

        merged: list[dict] = []
        for stop in cleaned:
            pos = float(np.clip(stop["pos"], 0.0, 1.0))
            color = [float(c) for c in stop["color"]]
            if merged and abs(pos - merged[-1]["pos"]) < self.gradient_min_spacing * 0.5:
                merged[-1] = {"pos": pos, "color": color}
            else:
                merged.append({"pos": pos, "color": color})

        if len(merged) == 1:
            merged.append({"pos": 1.0, "color": merged[0]["color"][:]})
            merged[0]["pos"] = 0.0

        if merged[0]["pos"] > self.gradient_min_spacing:
            merged.insert(0, {"pos": 0.0, "color": merged[0]["color"][:]})
        else:
            merged[0]["pos"] = 0.0

        if merged[-1]["pos"] < 1.0 - self.gradient_min_spacing:
            merged.append({"pos": 1.0, "color": merged[-1]["color"][:]})
        else:
            merged[-1]["pos"] = 1.0

        result: list[dict] = []
        for stop in merged:
            if (
                result
                and abs(stop["pos"] - result[-1]["pos"]) < self.gradient_min_spacing
                and np.allclose(stop["color"], result[-1]["color"], atol=1e-3)
            ):
                result[-1]["pos"] = stop["pos"]
                continue
            result.append({"pos": stop["pos"], "color": stop["color"][:]})

        return result

    def _stops_from_palette_colors(self, colors: np.ndarray) -> list[dict]:
        """Derive a manageable stop list from a palette color array."""
        if colors is None or len(colors) == 0:
            return [
                {"pos": 0.0, "color": [0.0, 0.0, 0.0]},
                {"pos": 1.0, "color": [1.0, 1.0, 1.0]},
            ]

        colors = np.asarray(colors, dtype=np.float32)
        if colors.ndim != 2:
            colors = colors.reshape(-1, 3)

        if colors.shape[1] > 3:
            colors = colors[:, :3]

        if float(np.max(colors)) > 1.0 + 1e-3:
            colors = colors / 255.0

        mask = ~np.isnan(colors).any(axis=1)
        colors = colors[mask]

        if len(colors) == 0:
            return [
                {"pos": 0.0, "color": [0.0, 0.0, 0.0]},
                {"pos": 1.0, "color": [1.0, 1.0, 1.0]},
            ]

        colors = np.clip(colors, 0.0, 1.0)

        if len(colors) > self.gradient_max_initial_stops:
            sampled = self._build_palette_lut_hsv(colors, size=self.gradient_max_initial_stops)
            positions = np.linspace(0.0, 1.0, len(sampled), dtype=np.float32)
            raw = [{"pos": float(pos), "color": sampled[idx].tolist()} for idx, pos in enumerate(positions)]
        else:
            positions = np.linspace(0.0, 1.0, len(colors), dtype=np.float32)
            raw = [{"pos": float(pos), "color": colors[idx].tolist()} for idx, pos in enumerate(positions)]

        return self._sanitize_stops(raw)

    def _init_gradient_from_palette(self, palette_name: str) -> None:
        """Initialize gradient stops from an existing palette."""
        saved_stops = self.gradient_metadata.get(palette_name)
        if saved_stops:
            stops = self._sanitize_stops(saved_stops)
        else:
            colors = _RUNTIME_PALETTES.get(palette_name)
            stops = self._stops_from_palette_colors(colors)

        self.gradient_stops.clear()
        self._next_stop_id = 0

        for stop in stops:
            entry = {
                "id": self._next_stop_id,
                "pos": float(stop["pos"]),
                "color": [
                    float(stop["color"][0]),
                    float(stop["color"][1]),
                    float(stop["color"][2]),
                ],
            }
            self._next_stop_id += 1
            self.gradient_stops.append(entry)

        self._sort_gradient_stops()
        self.selected_stop_id = self.gradient_stops[0]["id"] if self.gradient_stops else None
        self.dragging_stop_id = None
        self.custom_gradient_lut = self._build_gradient_lut()

    def _sort_gradient_stops(self) -> None:
        """Ensure gradient stops are ordered by position."""
        self.gradient_stops.sort(key=lambda stop: stop["pos"])

    def _build_gradient_lut(self, size: int = 256) -> np.ndarray:
        """Generate LUT samples from current gradient stops."""
        if not self.gradient_stops:
            return np.tile(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), (size, 1))

        stops = self.gradient_stops
        positions = np.array([stop["pos"] for stop in stops], dtype=np.float32)
        colors = np.array([stop["color"] for stop in stops], dtype=np.float32)

        if len(positions) == 1:
            return np.tile(colors[0], (size, 1))

        samples = np.linspace(0.0, 1.0, size, dtype=np.float32)
        lut = np.empty((size, 3), dtype=np.float32)
        for channel in range(3):
            lut[:, channel] = np.interp(samples, positions, colors[:, channel])
        return np.clip(lut, 0.0, 1.0)

    def _sample_gradient_color(self, position: float) -> list[float]:
        """Sample gradient color at arbitrary position."""
        lut = self._build_gradient_lut(size=512)
        idx = int(np.clip(position * (len(lut) - 1), 0, len(lut) - 1))
        return lut[idx].tolist()

    def _stop_to_x(self, position: float) -> float:
        bar_left = self.gradient_bar_padding
        bar_right = self.gradient_width - self.gradient_bar_padding
        return bar_left + position * (bar_right - bar_left)

    def _x_to_position(self, x: float) -> float:
        bar_left = self.gradient_bar_padding
        bar_right = self.gradient_width - self.gradient_bar_padding
        if bar_right <= bar_left:
            return 0.0
        return np.clip((x - bar_left) / (bar_right - bar_left), 0.0, 1.0)

    def _ensure_gradient_texture(self) -> None:
        """Create or update gradient preview texture."""
        if self.texture_registry_id is None:
            return

        lut = self.custom_gradient_lut if self.custom_gradient_lut is not None else self._build_gradient_lut()
        width = lut.shape[0]
        height = 4
        gradient = np.tile(lut[np.newaxis, :, :], (height, 1, 1))
        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[..., :3] = gradient
        data = rgba.reshape(-1)

        if self.gradient_texture_id is None:
            self.gradient_texture_id = dpg.add_dynamic_texture(
                width, height, data, parent=self.texture_registry_id
            )
        else:
            dpg.set_value(self.gradient_texture_id, data)

    def _update_gradient_drawlist(self) -> None:
        """Redraw gradient preview and handles."""
        if self.gradient_drawlist_id is None or not dpg.does_item_exist(self.gradient_drawlist_id):
            return

        self._ensure_gradient_texture()
        dpg.delete_item(self.gradient_drawlist_id, children_only=True)

        bar_left = self.gradient_bar_padding
        bar_right = self.gradient_width - self.gradient_bar_padding
        bar_top = self.gradient_bar_top
        bar_bottom = self.gradient_bar_bottom

        # Draw gradient bar
        if self.gradient_texture_id is not None:
            dpg.draw_image(
                self.gradient_texture_id,
                (bar_left, bar_top),
                (bar_right, bar_bottom),
                parent=self.gradient_drawlist_id,
            )
        dpg.draw_rectangle(
            (bar_left, bar_top),
            (bar_right, bar_bottom),
            color=(240, 240, 240, 220),
            thickness=1.0,
            parent=self.gradient_drawlist_id,
        )

        # Draw handles
        handle_base = bar_bottom + 4
        handle_height = 16
        self._handle_positions = {}

        for idx, stop in enumerate(self.gradient_stops):
            x = self._stop_to_x(stop["pos"])
            color = tuple(int(c * 255) for c in stop["color"])
            fill = (*color, 255)
            outline = (20, 20, 20, 255)
            if self.selected_stop_id == stop["id"]:
                outline = (255, 230, 120, 255)

            points = [
                (x, handle_base + handle_height),
                (x - self.gradient_handle_radius, handle_base),
                (x + self.gradient_handle_radius, handle_base),
            ]
            dpg.draw_triangle(
                points[0],
                points[1],
                points[2],
                color=outline,
                fill=fill,
                parent=self.gradient_drawlist_id,
            )

            # reference for hit testing
            self._handle_positions[stop["id"]] = (x, handle_base + handle_height / 2.0)

    def _select_stop_by_index(self, index: Optional[int]) -> None:
        if index is None or not (0 <= index < len(self.gradient_stops)):
            self.selected_stop_id = None
        else:
            self.selected_stop_id = self.gradient_stops[index]["id"]
        self._update_gradient_selection_ui()

    def _select_stop_by_id(self, stop_id: Optional[int]) -> None:
        if stop_id is None:
            self.selected_stop_id = None
        else:
            for idx, stop in enumerate(self.gradient_stops):
                if stop["id"] == stop_id:
                    self.selected_stop_id = stop_id
                    break
            else:
                self.selected_stop_id = None
        self._update_gradient_selection_ui()

    def _get_stop_index(self, stop_id: int) -> Optional[int]:
        for idx, stop in enumerate(self.gradient_stops):
            if stop["id"] == stop_id:
                return idx
        return None

    def _update_gradient_selection_ui(self) -> None:
        """Sync UI widgets with current selection."""
        if self.gradient_color_picker_id is None or not dpg.does_item_exist(self.gradient_color_picker_id):
            return

        has_selection = self.selected_stop_id is not None
        dpg.configure_item(self.gradient_color_picker_id, enabled=has_selection)
        if self.gradient_position_slider_id is not None and dpg.does_item_exist(self.gradient_position_slider_id):
            dpg.configure_item(self.gradient_position_slider_id, enabled=has_selection)
        if self.gradient_selected_text_id is not None and dpg.does_item_exist(self.gradient_selected_text_id):
            dpg.set_value(self.gradient_selected_text_id, "No stop selected")

        if not has_selection:
            if self.gradient_selected_text_id is not None and dpg.does_item_exist(self.gradient_selected_text_id):
                dpg.set_value(self.gradient_selected_text_id, "No stop selected")
            return

        idx = self._get_stop_index(self.selected_stop_id)
        if idx is None:
            if self.gradient_selected_text_id is not None and dpg.does_item_exist(self.gradient_selected_text_id):
                dpg.set_value(self.gradient_selected_text_id, "No stop selected")
            return

        stop = self.gradient_stops[idx]
        color = [float(stop["color"][0]), float(stop["color"][1]), float(stop["color"][2]), 1.0]
        dpg.set_value(self.gradient_color_picker_id, color)
        if self.gradient_position_slider_id is not None and dpg.does_item_exist(self.gradient_position_slider_id):
            dpg.set_value(self.gradient_position_slider_id, stop["pos"])
        if self.gradient_selected_text_id is not None and dpg.does_item_exist(self.gradient_selected_text_id):
            dpg.set_value(
                self.gradient_selected_text_id,
                f"Stop {idx + 1}/{len(self.gradient_stops)} @ {stop['pos']:.3f}",
            )

    def _set_stop_position(self, stop_id: int, position: float, apply: bool = True) -> None:
        idx = self._get_stop_index(stop_id)
        if idx is None:
            return

        min_pos = 0.0
        max_pos = 1.0
        padding = self.gradient_min_spacing
        if idx > 0:
            min_pos = self.gradient_stops[idx - 1]["pos"] + padding
        if idx < len(self.gradient_stops) - 1:
            max_pos = self.gradient_stops[idx + 1]["pos"] - padding
        if max_pos < min_pos:
            midpoint = (min_pos + max_pos) * 0.5
            min_pos = midpoint
            max_pos = midpoint

        clamped = np.clip(position, min_pos, max_pos)
        self.gradient_stops[idx]["pos"] = float(clamped)
        self._sort_gradient_stops()
        self._select_stop_by_id(stop_id)
        if apply:
            self._apply_gradient_change()

    def _apply_gradient_change(self, refresh_view: bool = True) -> None:
        self.custom_gradient_lut = self._build_gradient_lut()
        if refresh_view and self.gradient_drawlist_id is not None and dpg.does_item_exist(self.gradient_drawlist_id):
            self._update_gradient_drawlist()
            self._update_gradient_selection_ui()
        if self.use_custom_gradient:
            self._update_texture()

    def _add_gradient_stop(self, position: float, color: Optional[list[float]] = None) -> int:
        if color is None:
            color = self._sample_gradient_color(position)

        stop = {
            "id": self._next_stop_id,
            "pos": float(np.clip(position, 0.0, 1.0)),
            "color": [float(color[0]), float(color[1]), float(color[2])],
        }
        self._next_stop_id += 1
        self.gradient_stops.append(stop)
        self._sort_gradient_stops()
        self._select_stop_by_id(stop["id"])
        self._apply_gradient_change()
        return stop["id"]

    def _remove_stop(self, stop_id: int) -> None:
        if len(self.gradient_stops) <= 2:
            print("ℹ At least two stops required.")
            return

        idx = self._get_stop_index(stop_id)
        if idx is None:
            return

        del self.gradient_stops[idx]
        if self.gradient_stops:
            self._select_stop_by_index(min(idx, len(self.gradient_stops) - 1))
        else:
            self.selected_stop_id = None
        self._apply_gradient_change()

    def _hit_test_handle(self, local_x: float, local_y: float) -> Optional[int]:
        handle_base = self.gradient_bar_bottom + 4
        handle_top = handle_base
        handle_bottom = handle_base + 16
        for idx, stop in enumerate(self.gradient_stops):
            x = self._stop_to_x(stop["pos"])
            if abs(local_x - x) <= self.gradient_handle_radius * 1.2 and handle_top <= local_y <= handle_bottom:
                return idx
        return None

    def _gradient_local_coords(self, mouse_x: float, mouse_y: float) -> Optional[tuple[float, float]]:
        if self.gradient_drawlist_id is None or not dpg.does_item_exist(self.gradient_drawlist_id):
            return None
        rect_min = dpg.get_item_rect_min(self.gradient_drawlist_id)
        rect_max = dpg.get_item_rect_max(self.gradient_drawlist_id)
        if rect_min is None or rect_max is None:
            return None
        if not (rect_min[0] <= mouse_x <= rect_max[0] and rect_min[1] <= mouse_y <= rect_max[1]):
            return None
        return mouse_x - rect_min[0], mouse_y - rect_min[1]

    # ------------------------------------------------------------------
    # Gradient event handlers
    # ------------------------------------------------------------------

    def _on_gradient_mouse_down(self, sender, app_data) -> None:
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        coords = self._gradient_local_coords(mouse_x, mouse_y)
        if coords is None:
            return

        local_x, local_y = coords
        handle_index = self._hit_test_handle(local_x, local_y)
        if handle_index is not None:
            stop_id = self.gradient_stops[handle_index]["id"]
            self.dragging_stop_id = stop_id
            self._select_stop_by_id(stop_id)
            return

        # Add new stop when clicking inside gradient bar
        if self.gradient_bar_top <= local_y <= self.gradient_bar_bottom:
            position = self._x_to_position(local_x)
            color = self._sample_gradient_color(position)
            stop_id = self._add_gradient_stop(position, color=color)
            self.dragging_stop_id = stop_id

    def _on_gradient_mouse_drag(self, sender, app_data) -> None:
        if self.dragging_stop_id is None:
            return
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        coords = self._gradient_local_coords(mouse_x, mouse_y)
        if coords is None:
            return
        local_x, _ = coords
        position = self._x_to_position(local_x)
        self._set_stop_position(self.dragging_stop_id, position)

    def _on_gradient_mouse_release(self, sender, app_data) -> None:
        self.dragging_stop_id = None

    def _on_gradient_mouse_double_click(self, sender, app_data) -> None:
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        coords = self._gradient_local_coords(mouse_x, mouse_y)
        if coords is None:
            return
        local_x, local_y = coords
        handle_index = self._hit_test_handle(local_x, local_y)
        if handle_index is not None:
            stop_id = self.gradient_stops[handle_index]["id"]
            self._remove_stop(stop_id)

    def _on_gradient_color_change(self, sender, app_data) -> None:
        if self.selected_stop_id is None:
            return
        idx = self._get_stop_index(self.selected_stop_id)
        if idx is None:
            return
        # Dear PyGui can emit either [0,1] floats or [0,255] ints depending on configuration.
        if max(app_data[:3]) <= 1.0 + 1e-6:
            rgb = [float(app_data[i]) for i in range(3)]
        else:
            rgb = [float(app_data[i]) / 255.0 for i in range(3)]
        self.gradient_stops[idx]["color"] = rgb
        self._apply_gradient_change()

    def _on_gradient_position_change(self, sender, app_data) -> None:
        if self.selected_stop_id is None:
            return
        self._set_stop_position(self.selected_stop_id, float(app_data))

    def _on_use_gradient_toggle(self, sender, app_data) -> None:
        self.use_custom_gradient = bool(app_data)
        if self.use_custom_gradient:
            self._apply_gradient_change()
            dpg.set_value("current_palette_text", "Palette: Custom Gradient")
        else:
            dpg.set_value("current_palette_text", f"Palette: {self.current_palette}")
            self._update_texture()

    def _on_load_palette_into_gradient(self) -> None:
        self._init_gradient_from_palette(self.current_palette)
        self._apply_gradient_change()
        print(f"✓ Loaded '{self.current_palette}' into gradient editor")

    def _on_delete_selected_stop(self) -> None:
        if self.selected_stop_id is None:
            print("ℹ Select a stop to delete.")
            return
        self._remove_stop(self.selected_stop_id)

    def _on_add_midpoint_stop(self) -> None:
        self._add_gradient_stop(0.5)

    def _on_save_gradient(self) -> None:
        name = ""
        if self.gradient_name_input_id is not None:
            name = dpg.get_value(self.gradient_name_input_id).strip()
        if not name:
            print("✗ Enter a name before saving the gradient.")
            return
        colors = self._build_gradient_lut(size=16)
        add_palette(name, colors.tolist())
        self._record_gradient_metadata(name)
        print(f"✓ Saved custom gradient '{name}' to library")

        # Register colormap for new palette if not already present
        if name in _RUNTIME_PALETTES and name not in self.palette_colormaps:
            colors_normalized = _RUNTIME_PALETTES[name]
            colors_255 = [[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255] for c in colors_normalized]
            tag = f"colormap_{name.replace(' ', '_').replace('&', 'and')}"
            if self.colormap_registry_id is not None:
                dpg.add_colormap(colors_255, qualitative=False, tag=tag, parent=self.colormap_registry_id)
            self.palette_colormaps[name] = tag

        self.gradient_default_name = name
        if self.gradient_name_input_id is not None:
            dpg.set_value(self.gradient_name_input_id, name)

        # Switch to saved palette
        self.current_palette = name
        self.use_custom_gradient = False
        if self.use_gradient_checkbox_id is not None:
            dpg.set_value(self.use_gradient_checkbox_id, False)
        dpg.set_value("current_palette_text", f"Palette: {self.current_palette}")
        self._update_texture()

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
        self.use_custom_gradient = False
        if self.use_gradient_checkbox_id is not None:
            dpg.set_value(self.use_gradient_checkbox_id, False)
        self._init_gradient_from_palette(self.current_palette)
        self._apply_gradient_change()
        self._update_texture()
        dpg.set_value("current_palette_text", f"Palette: {self.current_palette}")
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

        # Reset custom gradient toggle and update editor with new palette
        self.use_custom_gradient = False
        if self.use_gradient_checkbox_id is not None:
            dpg.set_value(self.use_gradient_checkbox_id, False)
        self._init_gradient_from_palette(self.current_palette)
        self._apply_gradient_change()

        self._update_texture()
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

        with dpg.handler_registry() as handler:
            self.gradient_handler_id = handler
            dpg.add_mouse_down_handler(callback=self._on_gradient_mouse_down, button=dpg.mvMouseButton_Left)
            dpg.add_mouse_drag_handler(
                callback=self._on_gradient_mouse_drag,
                button=dpg.mvMouseButton_Left,
                threshold=0.0,
            )
            dpg.add_mouse_release_handler(callback=self._on_gradient_mouse_release, button=dpg.mvMouseButton_Left)
            dpg.add_mouse_double_click_handler(callback=self._on_gradient_mouse_double_click, button=dpg.mvMouseButton_Left)

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
        self._ensure_gradient_texture()

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
                    dpg.add_button(
                        label="Load Palette into Gradient",
                        width=200,
                        callback=lambda: self._on_load_palette_into_gradient(),
                    )
                    dpg.add_spacer(height=5)
                    dpg.add_separator()
                    dpg.add_spacer(height=5)
                    dpg.add_text("Custom Gradient Builder", color=(160, 220, 255))
                    self.use_gradient_checkbox_id = dpg.add_checkbox(
                        label="Use Custom Gradient",
                        default_value=self.use_custom_gradient,
                        callback=self._on_use_gradient_toggle,
                    )
                    self.gradient_selected_text_id = dpg.add_text("No stop selected")
                    with dpg.child_window(width=360, height=self.gradient_height + 50, no_scrollbar=True):
                        self.gradient_drawlist_id = dpg.add_drawlist(
                            width=self.gradient_width,
                            height=self.gradient_height,
                        )

                    self.gradient_color_picker_id = dpg.add_color_picker(
                        label="Stop Color",
                        default_value=[1.0, 1.0, 1.0, 1.0],
                        width=300,
                        no_alpha=True,
                        callback=self._on_gradient_color_change,
                    )
                    self.gradient_position_slider_id = dpg.add_slider_float(
                        label="Stop Position",
                        default_value=0.0,
                        min_value=0.0,
                        max_value=1.0,
                        callback=self._on_gradient_position_change,
                        width=250,
                        format="%.3f",
                    )
                    dpg.add_button(
                        label="Delete Stop",
                        width=200,
                        callback=lambda: self._on_delete_selected_stop(),
                    )
                    dpg.add_button(
                        label="Add Midpoint Stop",
                        width=200,
                        callback=lambda: self._on_add_midpoint_stop(),
                    )
                    dpg.add_spacer(height=10)
                    self.gradient_name_input_id = dpg.add_input_text(
                        label="Save As",
                        default_value=self.gradient_default_name,
                        width=220,
                    )
                    dpg.add_button(
                        label="💾 Save Gradient to Library",
                        width=220,
                        callback=lambda: self._on_save_gradient(),
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

        # Finalize gradient UI state
        self._update_gradient_drawlist()
        self._update_gradient_selection_ui()

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
