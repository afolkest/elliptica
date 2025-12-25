"""OKLCH palette creator with live project preview.

Create color palettes using OKLCH color space while seeing real-time
changes on an actual rendered Elliptica project.

Usage:
    python tools/oklch_palette_preview.py [path/to/project.elliptica]
"""

import sys
from pathlib import Path
from typing import Optional
import json

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from elliptica.serialization import load_project, load_render_cache
from elliptica.render import add_palette, USER_PALETTES_PATH
from elliptica.gpu import GPUContext
from elliptica.gpu.pipeline import build_base_rgb_gpu
from elliptica.gpu.ops import percentile_clip_gpu
from elliptica.colorspace.gamut import max_chroma_fast, gamut_map_to_srgb
from elliptica import defaults

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    print("Error: dearpygui not installed. Install with: pip install dearpygui")
    sys.exit(1)


class OklchPalettePreview:
    """OKLCH palette creator with live preview on rendered project."""

    def __init__(self, project_path: str):
        # Load project and render cache
        self.project_path = Path(project_path)
        self._load_project()
        self._init_preview_buffers()

        # Gradient stops: each is {id, pos, L, C, H}
        self.stops: list[dict] = []
        self._next_stop_id: int = 0
        self.selected_stop_id: Optional[int] = None
        self.is_dragging: bool = False

        # Gradient bar layout
        self.gradient_width = 500
        self.gradient_height = 70
        self.gradient_bar_top = 10
        self.gradient_bar_bottom = 45
        self.gradient_bar_padding = 25
        self.handle_radius = 9
        self.handle_hit_radius = 16
        self.handle_center_y = self.gradient_bar_bottom

        # Histogram layout
        self.hist_width = self.gradient_width
        self.hist_height = 70
        self.hist_bins = 128
        self.hist_padding = self.gradient_bar_padding
        self.hist_values: Optional[np.ndarray] = None

        # OKLCH picker dimensions
        self.slice_width = 360
        self.slice_height = 100
        self.c_max_absolute = 0.5
        self.use_relative_chroma = True
        self.c_max_display = self._get_c_display_max()
        self.chroma_interp_mix = 1.0

        # Precompute slice grids
        self._h_grid = np.linspace(0, 360, self.slice_width, endpoint=False, dtype=np.float32)
        self._c_grid = np.linspace(0, self.c_max_display, self.slice_height, dtype=np.float32)
        self._H_mesh, self._C_mesh = np.meshgrid(self._h_grid, self._c_grid)

        # Texture IDs
        self.texture_registry = None
        self.gradient_texture_id = None
        self.slice_texture_id = None
        self.preview_swatch_id = None
        self.l_gradient_texture_id = None
        self.image_texture_id = None
        self.histogram_drawlist_id = None

        # UI element IDs
        self.gradient_drawlist_id = None
        self.slice_drawlist_id = None
        self.slice_drag_active = False
        self.slider_block_theme = None

        # Display settings
        self.brightness = defaults.DEFAULT_BRIGHTNESS
        self.contrast = 1.0
        self.gamma = 1.0
        self.clip_percent = 1.5

        # Caches for palette and intensity processing
        self._palette_cache_key: Optional[tuple] = None
        self._lut_cache: dict[int, np.ndarray] = {}
        self._lut_tensor_cache: Optional[torch.Tensor] = None
        self._lut_tensor_cache_key: Optional[tuple] = None
        self._normalized_intensity_cache: Optional[np.ndarray] = None
        self._normalized_intensity_clip_percent: Optional[float] = None
        self._normalized_intensity_source_id: Optional[int] = None
        self._processed_intensity_cache: Optional[np.ndarray] = None
        self._processed_intensity_params: Optional[tuple] = None
        self._processed_intensity_source_id: Optional[int] = None
        self._normalized_tensor_cache: Optional[torch.Tensor] = None
        self._normalized_tensor_clip_percent: Optional[float] = None
        self._normalized_tensor_shape: Optional[tuple[int, int]] = None
        self._normalized_tensor_device: Optional[torch.device] = None

        # Persistence
        self.palettes_path = USER_PALETTES_PATH.with_name("oklch_palettes.json")
        self.saved_palettes = self._load_palettes()

        # Initialize with default gradient
        self._init_default_gradient()

        print(f"Loaded {self.width}x{self.height} render from {self.project_path.name}")
        print(f"GPU: {'enabled' if self.use_gpu else 'disabled'}")
        if self.preview_scale < 1.0:
            print(f"Preview: {self.display_width}x{self.display_height} (scale {self.preview_scale:.2f})")

    def _load_project(self):
        """Load project and render cache."""
        if not self.project_path.exists():
            raise FileNotFoundError(f"Project not found: {self.project_path}")

        self.state = load_project(str(self.project_path))

        # Try both cache extensions
        if self.project_path.suffix == '.elliptica':
            cache_paths = [
                self.project_path.with_suffix('.elliptica.cache'),
                self.project_path.with_suffix('.flowcol.cache'),
            ]
        else:
            cache_paths = [
                self.project_path.with_suffix('.flowcol.cache'),
                self.project_path.with_suffix('.elliptica.cache'),
            ]

        cache_path = None
        for p in cache_paths:
            if p.exists():
                cache_path = p
                break

        if cache_path is None:
            raise ValueError(f"No render cache found. Render the project first.")

        self.state.render_cache = load_render_cache(str(cache_path), self.state.project)
        if self.state.render_cache is None or self.state.render_cache.result is None:
            raise ValueError(f"Failed to load render cache from {cache_path}")

        self.lic_array = self.state.render_cache.result.array
        self.height, self.width = self.lic_array.shape

        self.use_gpu = GPUContext.is_available()
        if self.use_gpu:
            self.lic_tensor = GPUContext.to_gpu(self.lic_array)
        else:
            self.lic_tensor = None

        # Pre-allocate RGBA buffer
        self._rgba_buffer: Optional[np.ndarray] = None

        # Preview buffers (initialized after load)
        self.preview_scale = 1.0
        self.display_width = self.width
        self.display_height = self.height
        self.lic_preview_array: Optional[np.ndarray] = None
        self.lic_preview_tensor = None
        self.use_preview = True

    def _init_preview_buffers(self, max_display: int = 700) -> None:
        """Downsample LIC for fast UI preview updates."""
        if self.width == 0 or self.height == 0:
            return

        scale = min(1.0, max_display / max(self.width, self.height))
        self.preview_scale = scale
        self.display_width = max(1, int(self.width * scale))
        self.display_height = max(1, int(self.height * scale))

        # Keep full-res processing; only scale the on-screen display size.
        self.use_preview = False
        self.lic_preview_array = None
        self.lic_preview_tensor = None

    def _downsample_lic(self, lic: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize LIC array to preview resolution."""
        resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        lic_float = lic.astype(np.float32, copy=False)
        img = Image.fromarray(lic_float, mode="F")
        img = img.resize((width, height), resample=resample)
        return np.asarray(img, dtype=np.float32)

    def _get_lic_source(self) -> tuple[np.ndarray, Optional[torch.Tensor]]:
        """Return the LIC data used for previews."""
        if self.use_preview and self.lic_preview_array is not None:
            return self.lic_preview_array, self.lic_preview_tensor
        return self.lic_array, self.lic_tensor

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
        """Initialize with a two-stop gradient."""
        c0 = 0.08
        c1 = 0.06
        if self.use_relative_chroma:
            c0 = self._chroma_abs_to_rel(0.20, 270.0, c0)
            c1 = self._chroma_abs_to_rel(0.92, 60.0, c1)
        self.stops = [
            {"id": 0, "pos": 0.0, "L": 0.20, "C": c0, "H": 270.0},
            {"id": 1, "pos": 1.0, "L": 0.92, "C": c1, "H": 60.0},
        ]
        self._next_stop_id = 2
        self.selected_stop_id = 0

    def _sort_stops(self):
        self.stops.sort(key=lambda s: s["pos"])

    def _get_stop_index(self, stop_id: int) -> Optional[int]:
        for i, stop in enumerate(self.stops):
            if stop["id"] == stop_id:
                return i
        return None

    def _get_selected_stop(self) -> Optional[dict]:
        if self.selected_stop_id is None:
            return None
        idx = self._get_stop_index(self.selected_stop_id)
        return self.stops[idx] if idx is not None else None

    # ---------------------------------------------------------------
    # OKLCH utilities
    # ---------------------------------------------------------------

    def _oklch_to_rgb_clamped(self, L: float, C: float, H: float) -> tuple[float, float, float]:
        C_abs = self._chroma_to_abs(L, H, C)
        rgb = gamut_map_to_srgb(
            np.array(L, dtype=np.float32),
            np.array(C_abs, dtype=np.float32),
            np.array(H, dtype=np.float32),
            method='compress',
        )
        rgb = np.clip(rgb, 0.0, 1.0)
        return tuple(float(c) for c in rgb.flat)

    def _oklch_to_rgb255(self, L: float, C: float, H: float) -> tuple[int, int, int]:
        r, g, b = self._oklch_to_rgb_clamped(L, C, H)
        return (int(r * 255), int(g * 255), int(b * 255))

    def _get_max_chroma(self, L: float, H: float) -> float:
        return float(max_chroma_fast(
            np.array(L, dtype=np.float32),
            np.array(H, dtype=np.float32),
        ))

    def _is_in_gamut(self, L: float, C: float, H: float) -> bool:
        if self.use_relative_chroma:
            return C <= 1.0 + 1e-6
        return C <= self._get_max_chroma(L, H) + 1e-6

    def _get_c_display_max(self) -> float:
        return 1.0 if self.use_relative_chroma else self.c_max_absolute

    def _rebuild_slice_grids(self) -> None:
        self.c_max_display = self._get_c_display_max()
        self._c_grid = np.linspace(0, self.c_max_display, self.slice_height, dtype=np.float32)
        self._H_mesh, self._C_mesh = np.meshgrid(self._h_grid, self._c_grid)

    def _chroma_rel_to_abs(self, L: float, H: float, C_rel: float) -> float:
        max_c = self._get_max_chroma(L, H)
        if max_c <= 0.0:
            return 0.0
        return float(np.clip(C_rel, 0.0, 1.0) * max_c)

    def _chroma_rel_to_abs_array(
        self,
        L_arr: np.ndarray,
        H_arr: np.ndarray,
        C_rel_arr: np.ndarray,
    ) -> np.ndarray:
        max_c = max_chroma_fast(L_arr, H_arr)
        return np.clip(C_rel_arr, 0.0, 1.0) * max_c

    def _chroma_abs_to_rel(self, L: float, H: float, C_abs: float) -> float:
        max_c = self._get_max_chroma(L, H)
        if max_c <= 0.0:
            return 0.0
        return float(np.clip(C_abs / max_c, 0.0, 1.0))

    def _get_stop_chroma_rel_abs(self, stop: dict) -> tuple[float, float]:
        if self.use_relative_chroma:
            c_rel = float(stop["C"])
            c_abs = self._chroma_rel_to_abs(stop["L"], stop["H"], c_rel)
        else:
            c_abs = float(stop["C"])
            c_rel = self._chroma_abs_to_rel(stop["L"], stop["H"], c_abs)
        return c_rel, c_abs

    def _chroma_to_abs(self, L: float, H: float, C: float) -> float:
        if self.use_relative_chroma:
            return self._chroma_rel_to_abs(L, H, C)
        return float(max(C, 0.0))

    def _chroma_to_abs_array(
        self,
        L_arr: np.ndarray,
        H_arr: np.ndarray,
        C_arr: np.ndarray,
    ) -> np.ndarray:
        if self.use_relative_chroma:
            return self._chroma_rel_to_abs_array(L_arr, H_arr, C_arr)
        return np.clip(C_arr, 0.0, None)

    # ---------------------------------------------------------------
    # Gradient interpolation & LUT
    # ---------------------------------------------------------------

    def _get_palette_cache_key(self) -> tuple:
        return (
            float(self.chroma_interp_mix),
            tuple((float(s["pos"]), float(s["L"]), float(s["C"]), float(s["H"])) for s in self.stops),
        )

    def _get_lut(self, size: int = 256) -> np.ndarray:
        key = self._get_palette_cache_key()
        if self._palette_cache_key != key:
            self._palette_cache_key = key
            self._lut_cache.clear()
            self._lut_tensor_cache = None
            self._lut_tensor_cache_key = None

        lut = self._lut_cache.get(size)
        if lut is None:
            lut = self._build_lut(size=size)
            self._lut_cache[size] = lut
        return lut

    def _interpolate_oklch(self, t: float) -> tuple[float, float, float]:
        """Interpolate gradient at position t (0-1) in OKLCH space."""
        if not self.stops:
            return (0.5, 0.0, 0.0)

        if len(self.stops) == 1:
            s = self.stops[0]
            return (s["L"], s["C"], s["H"])

        t = np.clip(t, 0.0, 1.0)

        if t <= self.stops[0]["pos"]:
            s = self.stops[0]
            return (s["L"], s["C"], s["H"])

        if t >= self.stops[-1]["pos"]:
            s = self.stops[-1]
            return (s["L"], s["C"], s["H"])

        for i in range(len(self.stops) - 1):
            s0, s1 = self.stops[i], self.stops[i + 1]
            if s0["pos"] <= t <= s1["pos"]:
                frac = (t - s0["pos"]) / (s1["pos"] - s0["pos"])

                L = s0["L"] + frac * (s1["L"] - s0["L"])

                # Hue interpolation with wraparound
                h0, h1 = s0["H"], s1["H"]
                dh = h1 - h0
                if dh > 180:
                    dh -= 360
                elif dh < -180:
                    dh += 360
                H = (h0 + frac * dh) % 360

                c_rel0, c_abs0 = self._get_stop_chroma_rel_abs(s0)
                c_rel1, c_abs1 = self._get_stop_chroma_rel_abs(s1)
                c_rel = c_rel0 + frac * (c_rel1 - c_rel0)
                c_abs_rel = self._chroma_rel_to_abs(L, H, c_rel)
                c_abs = c_abs0 + frac * (c_abs1 - c_abs0)
                mix = float(np.clip(self.chroma_interp_mix, 0.0, 1.0))
                c_abs_mix = c_abs * (1.0 - mix) + c_abs_rel * mix

                if self.use_relative_chroma:
                    C = self._chroma_abs_to_rel(L, H, c_abs_mix)
                else:
                    C = c_abs_mix

                return (L, C, H)

        s = self.stops[-1]
        return (s["L"], s["C"], s["H"])

    def _build_lut(self, size: int = 256) -> np.ndarray:
        """Build RGB LUT from current gradient."""
        positions = np.linspace(0.0, 1.0, size, dtype=np.float32)

        if not self.stops:
            return np.zeros((size, 3), dtype=np.float32)

        if len(self.stops) == 1:
            stop = self.stops[0]
            L_arr = np.full(size, float(stop["L"]), dtype=np.float32)
            H_arr = np.full(size, float(stop["H"]), dtype=np.float32)
            c_rel, c_abs = self._get_stop_chroma_rel_abs(stop)
            C_rel_arr = np.full(size, float(c_rel), dtype=np.float32)
            C_abs_arr = np.full(size, float(c_abs), dtype=np.float32)
        else:
            stops = self.stops
            stop_count = len(stops)
            stop_c_rel = np.empty(stop_count, dtype=np.float32)
            stop_c_abs = np.empty(stop_count, dtype=np.float32)

            for i, stop in enumerate(stops):
                c_rel, c_abs = self._get_stop_chroma_rel_abs(stop)
                stop_c_rel[i] = c_rel
                stop_c_abs[i] = c_abs

            L_arr = np.full(size, float(stops[0]["L"]), dtype=np.float32)
            H_arr = np.full(size, float(stops[0]["H"]), dtype=np.float32)
            C_rel_arr = np.full(size, float(stop_c_rel[0]), dtype=np.float32)
            C_abs_arr = np.full(size, float(stop_c_abs[0]), dtype=np.float32)

            for i in range(stop_count - 1):
                s0, s1 = stops[i], stops[i + 1]
                p0, p1 = float(s0["pos"]), float(s1["pos"])
                if p1 <= p0:
                    continue

                mask = (positions >= p0) & (positions <= p1)
                if not np.any(mask):
                    continue

                frac = (positions[mask] - p0) / (p1 - p0)

                L_arr[mask] = s0["L"] + frac * (s1["L"] - s0["L"])

                h0, h1 = s0["H"], s1["H"]
                dh = h1 - h0
                if dh > 180:
                    dh -= 360
                elif dh < -180:
                    dh += 360
                H_arr[mask] = (h0 + frac * dh) % 360

                C_rel_arr[mask] = stop_c_rel[i] + frac * (stop_c_rel[i + 1] - stop_c_rel[i])
                C_abs_arr[mask] = stop_c_abs[i] + frac * (stop_c_abs[i + 1] - stop_c_abs[i])

            last = stops[-1]
            last_pos = float(last["pos"])
            tail_mask = positions > last_pos
            if np.any(tail_mask):
                c_rel, c_abs = self._get_stop_chroma_rel_abs(last)
                L_arr[tail_mask] = float(last["L"])
                H_arr[tail_mask] = float(last["H"])
                C_rel_arr[tail_mask] = float(c_rel)
                C_abs_arr[tail_mask] = float(c_abs)

        mix = float(np.clip(self.chroma_interp_mix, 0.0, 1.0))
        if mix <= 0.0:
            C_abs = np.clip(C_abs_arr, 0.0, None)
        elif mix >= 1.0:
            max_c = max_chroma_fast(L_arr, H_arr)
            C_abs = np.clip(C_rel_arr, 0.0, 1.0) * max_c
        else:
            max_c = max_chroma_fast(L_arr, H_arr)
            C_abs_rel = np.clip(C_rel_arr, 0.0, 1.0) * max_c
            C_abs = (1.0 - mix) * C_abs_arr + mix * C_abs_rel
            C_abs = np.clip(C_abs, 0.0, None)

        rgb = gamut_map_to_srgb(L_arr, C_abs, H_arr, method='compress')
        return np.clip(rgb, 0.0, 1.0)

    # ---------------------------------------------------------------
    # Apply palette to image
    # ---------------------------------------------------------------

    def _normalize_unit(self, arr: np.ndarray) -> np.ndarray:
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_max > arr_min:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr, dtype=np.float32)

    def _get_normalized_intensity_cpu(self, lic: np.ndarray) -> np.ndarray:
        clip_percent = float(self.clip_percent)
        source_id = id(lic)
        if (
            self._normalized_intensity_cache is not None
            and self._normalized_intensity_clip_percent is not None
            and abs(self._normalized_intensity_clip_percent - clip_percent) < 1e-6
            and self._normalized_intensity_source_id == source_id
        ):
            return self._normalized_intensity_cache

        arr = lic.astype(np.float32, copy=False)
        if clip_percent > 0.0:
            vmin = float(np.percentile(arr, clip_percent))
            vmax = float(np.percentile(arr, 100.0 - clip_percent))
            if vmax > vmin:
                norm = (arr - vmin) / (vmax - vmin)
            else:
                norm = self._normalize_unit(arr)
        else:
            norm = self._normalize_unit(arr)

        norm = np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)

        self._normalized_intensity_cache = norm
        self._normalized_intensity_clip_percent = clip_percent
        self._normalized_intensity_source_id = source_id
        self._processed_intensity_params = None
        return norm

    def _process_intensity_cpu(self, lic: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply clip/brightness/contrast/gamma to LIC for histogram and CPU color."""
        base = self._get_normalized_intensity_cpu(lic)
        params = (float(self.clip_percent), float(self.brightness), float(self.contrast), float(self.gamma))
        source_id = self._normalized_intensity_source_id
        if (
            self._processed_intensity_cache is not None
            and self._processed_intensity_params == params
            and self._processed_intensity_source_id == source_id
        ):
            return self._processed_intensity_cache, False

        if self._processed_intensity_cache is None or self._processed_intensity_cache.shape != base.shape:
            self._processed_intensity_cache = np.empty_like(base, dtype=np.float32)

        arr = self._processed_intensity_cache
        np.copyto(arr, base)

        if self.contrast != 1.0:
            arr -= 0.5
            arr *= self.contrast
            arr += 0.5
            np.clip(arr, 0.0, 1.0, out=arr)

        if self.brightness != 0.0:
            arr += self.brightness
            np.clip(arr, 0.0, 1.0, out=arr)

        gamma = max(float(self.gamma), 1e-3)
        if gamma != 1.0:
            np.power(arr, gamma, out=arr)
            np.clip(arr, 0.0, 1.0, out=arr)

        self._processed_intensity_params = params
        self._processed_intensity_source_id = source_id
        return arr, True

    def _get_normalized_tensor(self, lic_tensor: torch.Tensor) -> torch.Tensor:
        clip_percent = float(self.clip_percent)
        shape = tuple(lic_tensor.shape)
        device = lic_tensor.device
        if (
            self._normalized_tensor_cache is not None
            and self._normalized_tensor_clip_percent is not None
            and abs(self._normalized_tensor_clip_percent - clip_percent) < 1e-6
            and self._normalized_tensor_shape == shape
            and self._normalized_tensor_device == device
        ):
            return self._normalized_tensor_cache

        normalized, _, _ = percentile_clip_gpu(lic_tensor, clip_percent)
        self._normalized_tensor_cache = normalized
        self._normalized_tensor_clip_percent = clip_percent
        self._normalized_tensor_shape = shape
        self._normalized_tensor_device = device
        return normalized

    def _apply_palette(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Apply current gradient to LIC array.

        Returns:
            (RGB float image in [0,1], processed intensity array in [0,1], intensity_changed)
        """
        lut_numpy = self._get_lut(size=256)

        lic_array, lic_tensor = self._get_lic_source()
        processed_intensity, intensity_changed = self._process_intensity_cpu(lic_array)

        if self.use_gpu and lic_tensor is not None:
            lut_tensor = self._lut_tensor_cache
            if lut_tensor is None or self._lut_tensor_cache_key != self._palette_cache_key:
                lut_tensor = GPUContext.to_gpu(lut_numpy)
                self._lut_tensor_cache = lut_tensor
                self._lut_tensor_cache_key = self._palette_cache_key

            normalized_tensor = self._get_normalized_tensor(lic_tensor)
            rgb_tensor = build_base_rgb_gpu(
                lic_tensor,
                clip_percent=self.clip_percent,
                brightness=self.brightness,
                contrast=self.contrast,
                gamma=self.gamma,
                color_enabled=True,
                lut=lut_tensor,
                normalized_tensor=normalized_tensor,
            )
            rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)
            rgb_array = GPUContext.to_cpu(rgb_tensor)
        else:
            lut = lut_numpy
            indices = (processed_intensity * (len(lut) - 1)).astype(np.int32)
            indices = np.clip(indices, 0, len(lut) - 1)
            rgb_array = lut[indices]

        return rgb_array.astype(np.float32, copy=False), processed_intensity, intensity_changed

    # ---------------------------------------------------------------
    # Texture generation
    # ---------------------------------------------------------------

    def _generate_gradient_texture(self) -> np.ndarray:
        """Generate gradient bar texture."""
        lut = self._get_lut(size=self.gradient_width)
        height = 28
        rgba = np.ones((height, self.gradient_width, 4), dtype=np.float32)
        rgba[:, :, :3] = lut[np.newaxis, :, :]
        return rgba

    def _generate_ch_slice(self, L: float) -> np.ndarray:
        """Generate C x H slice at fixed L."""
        L_arr = np.full_like(self._C_mesh, L)
        max_c = max_chroma_fast(L_arr, self._H_mesh)

        if self.use_relative_chroma:
            C_abs = self._C_mesh * max_c
            rgb = gamut_map_to_srgb(L_arr, C_abs, self._H_mesh, method='clip')
            rgba = np.ones((self.slice_height, self.slice_width, 4), dtype=np.float32)
            rgba[..., :3] = np.clip(rgb, 0.0, 1.0)
            return rgba

        rgb = gamut_map_to_srgb(L_arr, self._C_mesh, self._H_mesh, method='clip')

        rgba = np.ones((self.slice_height, self.slice_width, 4), dtype=np.float32)
        rgba[..., :3] = np.clip(rgb, 0.0, 1.0)

        out_of_gamut = self._C_mesh > max_c
        rgba[out_of_gamut, :3] = 0.15
        rgba[out_of_gamut, 3] = 0.5

        return rgba

    def _generate_l_gradient(self, C: float, H: float) -> np.ndarray:
        """Generate L gradient bar."""
        width = 360
        height = 14

        L_arr = np.linspace(0, 1, width, dtype=np.float32)
        H_arr = np.full(width, H, dtype=np.float32)

        max_c = max_chroma_fast(L_arr, H_arr)
        if self.use_relative_chroma:
            C_rel = float(np.clip(C, 0.0, 1.0))
            C_arr = max_c * C_rel

            rgb = gamut_map_to_srgb(L_arr, C_arr, H_arr, method='clip')
            rgb = np.clip(rgb, 0.0, 1.0)

            rgba = np.ones((height, width, 4), dtype=np.float32)
            rgba[:, :, :3] = rgb[np.newaxis, :, :]
            return rgba

        C_arr = np.full(width, float(np.clip(C, 0.0, self.c_max_absolute)), dtype=np.float32)
        in_gamut = C_arr <= max_c

        rgb = gamut_map_to_srgb(L_arr, C_arr, H_arr, method='clip')
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb[~in_gamut] = [0.2, 0.2, 0.2]

        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = rgb[np.newaxis, :, :]
        return rgba

    def _generate_preview_swatch(self, L: float, C: float, H: float) -> np.ndarray:
        """Generate preview swatch."""
        size = 50
        r, g, b = self._oklch_to_rgb_clamped(L, C, H)
        rgba = np.ones((size, size, 4), dtype=np.float32)
        rgba[..., :3] = [r, g, b]
        return rgba

    # ---------------------------------------------------------------
    # Texture updates
    # ---------------------------------------------------------------

    def _update_gradient_texture(self):
        rgba = self._generate_gradient_texture()
        data = rgba.ravel()
        if self.gradient_texture_id is None:
            self.gradient_texture_id = dpg.add_dynamic_texture(
                self.gradient_width, 28, data, parent=self.texture_registry
            )
        else:
            dpg.set_value(self.gradient_texture_id, data)

    def _update_slice_texture(self):
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
        stop = self._get_selected_stop()
        C = stop["C"] if stop else 0.0
        H = stop["H"] if stop else 0.0

        rgba = self._generate_l_gradient(C, H)
        data = rgba.ravel()
        if self.l_gradient_texture_id is None:
            self.l_gradient_texture_id = dpg.add_dynamic_texture(
                360, 14, data, parent=self.texture_registry
            )
        else:
            dpg.set_value(self.l_gradient_texture_id, data)

    def _update_preview_swatch(self):
        stop = self._get_selected_stop()
        L = stop["L"] if stop else 0.5
        C = stop["C"] if stop else 0.0
        H = stop["H"] if stop else 0.0

        rgba = self._generate_preview_swatch(L, C, H)
        data = rgba.ravel()
        if self.preview_swatch_id is None:
            self.preview_swatch_id = dpg.add_dynamic_texture(
                50, 50, data, parent=self.texture_registry
            )
        else:
            dpg.set_value(self.preview_swatch_id, data)

    def _update_image_texture(self):
        """Update the main preview image and histogram."""
        rgb_float, processed_intensity, intensity_changed = self._apply_palette()
        height, width = rgb_float.shape[:2]

        if self._rgba_buffer is None or self._rgba_buffer.shape[:2] != (height, width):
            self._rgba_buffer = np.empty((height, width, 4), dtype=np.float32)

        self._rgba_buffer[:, :, :3] = rgb_float
        self._rgba_buffer[:, :, 3] = 1.0
        data = self._rgba_buffer.ravel()

        if self.image_texture_id is None:
            self.image_texture_id = dpg.add_dynamic_texture(
                width, height, data, parent=self.texture_registry
            )
        else:
            dpg.set_value(self.image_texture_id, data)

        if intensity_changed:
            self._update_histogram_from_processed(processed_intensity)

    def _update_histogram_from_processed(self, processed: np.ndarray):
        """Compute histogram from processed intensity and refresh drawlist."""
        counts, _ = np.histogram(processed, bins=self.hist_bins, range=(0.0, 1.0))
        counts = counts.astype(np.float32)
        if counts.max() > 0:
            counts = counts / counts.max()
        self.hist_values = counts
        self._update_histogram_drawlist()

    def _update_histogram_drawlist(self):
        """Draw histogram bars."""
        if self.histogram_drawlist_id is None:
            return
        dpg.delete_item(self.histogram_drawlist_id, children_only=True)

        if self.hist_values is None:
            return

        bar_left = self.hist_padding
        bar_top = 0
        bar_bottom = self.hist_height - 2
        inner_width = max(1.0, self.hist_width - 2 * self.hist_padding)
        bin_w = inner_width / self.hist_bins

        dpg.draw_rectangle(
            (0, bar_top),
            (self.hist_width, bar_bottom),
            color=(80, 80, 80, 180),
            thickness=1,
            parent=self.histogram_drawlist_id,
        )

        for i, v in enumerate(self.hist_values):
            x0 = bar_left + i * bin_w
            x1 = x0 + bin_w
            y1 = bar_bottom
            y0 = bar_bottom - (v * (self.hist_height - 4))
            dpg.draw_rectangle(
                (x0, y0),
                (x1, y1),
                fill=(180, 180, 190, 180),
                color=(0, 0, 0, 0),
                parent=self.histogram_drawlist_id,
            )

    # ---------------------------------------------------------------
    # Gradient bar drawing
    # ---------------------------------------------------------------

    def _stop_to_x(self, pos: float) -> float:
        bar_left = self.gradient_bar_padding
        bar_right = self.gradient_width - self.gradient_bar_padding
        return bar_left + pos * (bar_right - bar_left)

    def _x_to_pos(self, x: float) -> float:
        bar_left = self.gradient_bar_padding
        bar_right = self.gradient_width - self.gradient_bar_padding
        return float(np.clip((x - bar_left) / (bar_right - bar_left), 0.0, 1.0))

    def _update_gradient_drawlist(self):
        if self.gradient_drawlist_id is None:
            return

        dpg.delete_item(self.gradient_drawlist_id, children_only=True)

        bar_left = self.gradient_bar_padding
        bar_right = self.gradient_width - self.gradient_bar_padding

        dpg.draw_image(
            self.gradient_texture_id,
            (bar_left, self.gradient_bar_top),
            (bar_right, self.gradient_bar_bottom),
            parent=self.gradient_drawlist_id,
        )

        dpg.draw_rectangle(
            (bar_left, self.gradient_bar_top),
            (bar_right, self.gradient_bar_bottom),
            color=(100, 100, 100, 255),
            thickness=1,
            parent=self.gradient_drawlist_id,
        )

        # Draw handles
        for is_selected_pass in [False, True]:
            for stop in self.stops:
                is_selected = stop["id"] == self.selected_stop_id
                if is_selected != is_selected_pass:
                    continue

                x = self._stop_to_x(stop["pos"])
                y = self.handle_center_y
                r, g, b = self._oklch_to_rgb255(stop["L"], stop["C"], stop["H"])

                line_color = (255, 255, 255, 100) if is_selected else (255, 255, 255, 40)
                dpg.draw_line(
                    (x, self.gradient_bar_top + 2),
                    (x, self.gradient_bar_bottom - 2),
                    color=line_color,
                    thickness=1,
                    parent=self.gradient_drawlist_id,
                )

                if is_selected:
                    dpg.draw_circle(
                        (x, y), self.handle_radius + 3,
                        color=(255, 200, 80, 180),
                        thickness=2,
                        parent=self.gradient_drawlist_id,
                    )

                dpg.draw_circle(
                    (x, y), self.handle_radius,
                    color=(255, 255, 255, 255),
                    fill=(255, 255, 255, 255),
                    parent=self.gradient_drawlist_id,
                )

                dpg.draw_circle(
                    (x, y), self.handle_radius - 1,
                    color=(r, g, b, 255),
                    fill=(r, g, b, 255),
                    parent=self.gradient_drawlist_id,
                )

    def _update_slice_crosshair(self):
        if self.slice_drawlist_id is None:
            return

        dpg.delete_item(self.slice_drawlist_id, children_only=True)

        dpg.draw_image(
            self.slice_texture_id,
            (0, 0),
            (self.slice_width, self.slice_height),
            parent=self.slice_drawlist_id,
        )

        stop = self._get_selected_stop()
        if stop is None:
            return

        x = stop["H"]
        y = np.clip(stop["C"] / self.c_max_display, 0.0, 1.0) * self.slice_height

        color = (255, 255, 255, 200)
        dpg.draw_line((0, y), (self.slice_width, y), color=color, thickness=1,
                     parent=self.slice_drawlist_id)
        dpg.draw_line((x, 0), (x, self.slice_height), color=color, thickness=1,
                     parent=self.slice_drawlist_id)
        dpg.draw_circle((x, y), 4, color=color, thickness=2, parent=self.slice_drawlist_id)

    # ---------------------------------------------------------------
    # UI sync
    # ---------------------------------------------------------------

    def _sync_chroma_controls(self, stop: Optional[dict] = None):
        if not dpg.does_item_exist("c_slider"):
            return

        max_value = self._get_c_display_max()
        format_str = "%.3f"
        dpg.configure_item("c_slider", max_value=max_value, format=format_str)

        if stop is not None:
            c_val = float(np.clip(stop["C"], 0.0, max_value))
            dpg.set_value("c_slider", c_val)

        if dpg.does_item_exist("chroma_mode_toggle"):
            dpg.set_value("chroma_mode_toggle", self.use_relative_chroma)
        if dpg.does_item_exist("chroma_interp_button"):
            label = "Interp: Rel" if self.chroma_interp_mix >= 0.5 else "Interp: Abs"
            dpg.configure_item("chroma_interp_button", label=label)

    def _sync_ui_from_stop(self):
        stop = self._get_selected_stop()
        if stop is None:
            dpg.configure_item("l_slider", enabled=False)
            dpg.configure_item("c_slider", enabled=False)
            dpg.configure_item("h_slider", enabled=False)
            if dpg.does_item_exist("chroma_mode_toggle"):
                dpg.configure_item("chroma_mode_toggle", enabled=False)
            dpg.set_value("stop_info_text", "No stop selected")
            return

        dpg.configure_item("l_slider", enabled=True)
        dpg.configure_item("c_slider", enabled=True)
        dpg.configure_item("h_slider", enabled=True)
        if dpg.does_item_exist("chroma_mode_toggle"):
            dpg.configure_item("chroma_mode_toggle", enabled=True)

        dpg.set_value("l_slider", stop["L"])
        dpg.set_value("h_slider", stop["H"])
        self._sync_chroma_controls(stop)

        idx = self._get_stop_index(stop["id"])
        r, g, b = self._oklch_to_rgb255(stop["L"], stop["C"], stop["H"])
        in_gamut = self._is_in_gamut(stop["L"], stop["C"], stop["H"])
        gamut_str = "" if in_gamut else " [!]"

        dpg.set_value("stop_info_text",
            f"Stop {idx + 1}/{len(self.stops)} @ {stop['pos']:.2f}{gamut_str}"
        )

    def _refresh_all(self):
        """Full refresh of all visuals including image."""
        self._mark_dirty(full=True)

    def _refresh_picker_only(self):
        """Refresh picker visuals without image update."""
        self._mark_dirty(full=True)

    def _refresh_gradient_and_image(self):
        """Refresh gradient bar and image (after color change)."""
        self._mark_dirty(full=True)

    # ---------------------------------------------------------------
    # Event handlers
    # ---------------------------------------------------------------

    def _hit_test_handle(self, local_x: float, local_y: float) -> Optional[int]:
        best_idx = None
        best_dist = float('inf')

        for idx, stop in enumerate(self.stops):
            x = self._stop_to_x(stop["pos"])
            y = self.handle_center_y
            dist = ((local_x - x) ** 2 + (local_y - y) ** 2) ** 0.5

            if dist <= self.handle_hit_radius and dist < best_dist:
                best_dist = dist
                best_idx = idx

        return best_idx

    def _gradient_local_coords(self, mouse_x: float, mouse_y: float, clamp: bool = False):
        if not dpg.does_item_exist(self.gradient_drawlist_id):
            return None
        rect_min = dpg.get_item_rect_min(self.gradient_drawlist_id)
        rect_max = dpg.get_item_rect_max(self.gradient_drawlist_id)
        if rect_min is None or rect_max is None:
            return None

        local_x = mouse_x - rect_min[0]
        local_y = mouse_y - rect_min[1]

        if clamp:
            local_x = max(0, min(local_x, rect_max[0] - rect_min[0]))
            local_y = max(0, min(local_y, rect_max[1] - rect_min[1]))
            return local_x, local_y
        else:
            if not (rect_min[0] <= mouse_x <= rect_max[0] and rect_min[1] <= mouse_y <= rect_max[1]):
                return None
            return local_x, local_y

    def _on_gradient_mouse_down(self, sender, app_data):
        self.is_dragging = False
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)

        if not dpg.does_item_exist(self.gradient_drawlist_id):
            return
        rect_min = dpg.get_item_rect_min(self.gradient_drawlist_id)
        rect_max = dpg.get_item_rect_max(self.gradient_drawlist_id)
        if rect_min is None or rect_max is None:
            return

        expanded_bottom = rect_max[1] + self.handle_radius
        if not (rect_min[0] <= mouse_x <= rect_max[0] and rect_min[1] <= mouse_y <= expanded_bottom):
            return

        local_x = mouse_x - rect_min[0]
        local_y = mouse_y - rect_min[1]

        handle_idx = self._hit_test_handle(local_x, local_y)

        if handle_idx is None:
            self.is_dragging = False
            return

        self.selected_stop_id = self.stops[handle_idx]["id"]
        self.is_dragging = True
        self._refresh_all()

    def _on_gradient_mouse_drag(self, sender, app_data):
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            self.is_dragging = False
            return
        if not self.is_dragging or self.selected_stop_id is None:
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        coords = self._gradient_local_coords(mouse_x, mouse_y, clamp=True)
        if coords is None:
            return

        local_x, _ = coords
        new_pos = self._x_to_pos(local_x)

        idx = self._get_stop_index(self.selected_stop_id)
        if idx is not None:
            min_pos = self.stops[idx - 1]["pos"] + 0.005 if idx > 0 else 0.0
            max_pos = self.stops[idx + 1]["pos"] - 0.005 if idx < len(self.stops) - 1 else 1.0
            self.stops[idx]["pos"] = float(np.clip(new_pos, min_pos, max_pos))

            self._mark_dirty(full=True)

    def _on_gradient_mouse_release(self, sender, app_data):
        if self.is_dragging:
            self.is_dragging = False
            self._sort_stops()
            self._mark_dirty(full=True)

    def _on_gradient_double_click(self, sender, app_data):
        if self.is_dragging:
            self.is_dragging = False
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)

        if not dpg.does_item_exist(self.gradient_drawlist_id):
            return
        rect_min = dpg.get_item_rect_min(self.gradient_drawlist_id)
        rect_max = dpg.get_item_rect_max(self.gradient_drawlist_id)
        if rect_min is None or rect_max is None:
            return

        expanded_bottom = rect_max[1] + self.handle_radius
        if not (rect_min[0] <= mouse_x <= rect_max[0] and rect_min[1] <= mouse_y <= expanded_bottom):
            return

        local_x = mouse_x - rect_min[0]
        local_y = mouse_y - rect_min[1]

        handle_idx = self._hit_test_handle(local_x, local_y)
        if handle_idx is not None:
            if len(self.stops) > 2:
                self._remove_stop(self.stops[handle_idx]["id"])
        elif self.gradient_bar_top <= local_y <= self.gradient_bar_bottom + self.handle_radius:
            pos = self._x_to_pos(local_x)
            L, C, H = self._interpolate_oklch(pos)
            self._add_stop(pos, L, C, H)

    def _on_slice_pointer(self, sender, app_data):
        """Handle pointer down/drag within the CxH slice."""
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            self.slice_drag_active = False
            return

        if not dpg.does_item_exist(self.slice_drawlist_id):
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min(self.slice_drawlist_id)
        rect_max = dpg.get_item_rect_max(self.slice_drawlist_id)
        if rect_min is None or rect_max is None:
            return

        inside = rect_min[0] <= mouse_x <= rect_max[0] and rect_min[1] <= mouse_y <= rect_max[1]
        if not self.slice_drag_active and not inside:
            return

        self.slice_drag_active = True

        # Clamp to bounds so slight drift outside keeps the drag smooth
        local_x = np.clip(mouse_x - rect_min[0], 0, rect_max[0] - rect_min[0])
        local_y = np.clip(mouse_y - rect_min[1], 0, rect_max[1] - rect_min[1])

        stop = self._get_selected_stop()
        if stop is None:
            return

        new_h = float(np.clip(local_x, 0, 359.9))
        new_c = float(np.clip((local_y / self.slice_height) * self.c_max_display, 0.0, 1.0))

        stop["H"] = new_h
        stop["C"] = new_c

        dpg.set_value("h_slider", new_h)
        dpg.set_value("c_slider", new_c)

        self._mark_dirty(full=True)

    def _on_l_change(self, sender, value):
        stop = self._get_selected_stop()
        if stop is None:
            return
        stop["L"] = value
        self._mark_dirty(full=True)

    def _on_c_change(self, sender, value):
        stop = self._get_selected_stop()
        if stop is None:
            return
        stop["C"] = float(np.clip(value, 0.0, self._get_c_display_max()))
        self._mark_dirty(full=True)

    def _on_h_change(self, sender, value):
        stop = self._get_selected_stop()
        if stop is None:
            return
        stop["H"] = value
        self._mark_dirty(full=True)

    def _on_chroma_mode_change(self, sender, value):
        new_relative = bool(value)
        if new_relative == self.use_relative_chroma:
            return

        if new_relative:
            for stop in self.stops:
                stop["C"] = self._chroma_abs_to_rel(stop["L"], stop["H"], stop["C"])
        else:
            for stop in self.stops:
                stop["C"] = self._chroma_rel_to_abs(stop["L"], stop["H"], stop["C"])

        self.use_relative_chroma = new_relative
        self._rebuild_slice_grids()
        self._sync_chroma_controls(self._get_selected_stop())
        self._mark_dirty(full=True)

    def _on_chroma_interp_toggle(self, sender, app_data=None):
        self.chroma_interp_mix = 0.0 if self.chroma_interp_mix >= 0.5 else 1.0
        if dpg.does_item_exist("chroma_interp_button"):
            label = "Interp: Rel" if self.chroma_interp_mix >= 0.5 else "Interp: Abs"
            dpg.configure_item("chroma_interp_button", label=label)
        self._mark_dirty(full=True)

    def _on_brightness_change(self, sender, value):
        self.brightness = value
        self._mark_dirty(full=True)

    def _on_contrast_change(self, sender, value):
        self.contrast = value
        self._mark_dirty(full=True)

    def _on_gamma_change(self, sender, value):
        self.gamma = value
        self._mark_dirty(full=True)

    def _on_clip_change(self, sender, value):
        self.clip_percent = max(0.0, value)
        self._mark_dirty(full=True)

    # ---------------------------------------------------------------
    # Stop management
    # ---------------------------------------------------------------

    def _add_stop(self, pos: float, L: float, C: float, H: float) -> int:
        stop = {"id": self._next_stop_id, "pos": pos, "L": L, "C": C, "H": H}
        self._next_stop_id += 1
        self.stops.append(stop)
        self._sort_stops()
        self.selected_stop_id = stop["id"]
        self._mark_dirty(full=True)
        return stop["id"]

    def _remove_stop(self, stop_id: int):
        if len(self.stops) <= 2:
            return

        idx = self._get_stop_index(stop_id)
        if idx is None:
            return

        del self.stops[idx]

        if self.stops:
            new_idx = min(idx, len(self.stops) - 1)
            self.selected_stop_id = self.stops[new_idx]["id"]
        else:
            self.selected_stop_id = None

        self._mark_dirty(full=True)

    def _on_delete_stop(self):
        if self.selected_stop_id is not None:
            self._remove_stop(self.selected_stop_id)

    def _on_add_stop(self):
        L, C, H = self._interpolate_oklch(0.5)
        self._add_stop(0.5, L, C, H)

    def _on_clamp_to_gamut(self):
        stop = self._get_selected_stop()
        if stop is None:
            return
        if self.use_relative_chroma:
            if stop["C"] > 1.0:
                stop["C"] = 1.0
                dpg.set_value("c_slider", 1.0)
                self._mark_dirty(full=True)
            return

        max_c = self._get_max_chroma(stop["L"], stop["H"])
        if stop["C"] > max_c:
            stop["C"] = max_c
            dpg.set_value("c_slider", max_c)
            self._mark_dirty(full=True)

    # ---------------------------------------------------------------
    # Save/Load
    # ---------------------------------------------------------------

    def _on_save_palette(self):
        name = dpg.get_value("palette_name_input").strip()
        if not name:
            print("Enter a name first")
            return

        stops_data = [
            {
                "pos": s["pos"],
                "L": s["L"],
                "C": (
                    self._chroma_rel_to_abs(s["L"], s["H"], s["C"])
                    if self.use_relative_chroma
                    else s["C"]
                ),
                "H": s["H"],
            }
            for s in self.stops
        ]
        self.saved_palettes[name] = stops_data
        self._save_palettes()

        lut = self._get_lut(size=16)
        add_palette(name, lut.tolist())

        print(f"Saved palette '{name}'")

    def _on_load_palette(self, sender, app_data, user_data):
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
                "C": (
                    self._chroma_abs_to_rel(s["L"], s["H"], s["C"])
                    if self.use_relative_chroma
                    else s["C"]
                ),
                "H": s["H"],
            })
            self._next_stop_id += 1

        self._sort_stops()
        self.selected_stop_id = self.stops[0]["id"] if self.stops else None
        dpg.set_value("palette_name_input", name)
        self._mark_dirty(full=True)

    def _on_new_palette(self):
        self._init_default_gradient()
        dpg.set_value("palette_name_input", "")
        self._mark_dirty(full=True)

    # ---------------------------------------------------------------
    # UI build
    # ---------------------------------------------------------------

    def _build_ui(self):
        dpg.create_context()

        self.texture_registry = dpg.add_texture_registry()

        if self.slider_block_theme is None:
            self.slider_block_theme = dpg.add_theme()
            with dpg.theme_component(dpg.mvAll, parent=self.slider_block_theme):
                dpg.add_theme_style(
                    dpg.mvStyleVar_ItemSpacing,
                    8,
                    2,
                    category=dpg.mvThemeCat_Core,
                )

        # Initialize textures
        self._update_gradient_texture()
        self._update_slice_texture()
        self._update_l_gradient_texture()
        self._update_preview_swatch()
        self._update_image_texture()

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

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=self._on_slice_pointer)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,
                                       callback=self._on_slice_pointer, threshold=0.0)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                          callback=lambda s, a: setattr(self, "slice_drag_active", False))

        # Calculate display size
        display_w = self.display_width
        display_h = self.display_height

        panel_width = max(520, self.gradient_width + 40)

        # Main window
        with dpg.window(label="OKLCH Palette Preview", tag="main_window"):
            with dpg.group(horizontal=True):
                # Left: Preview image
                with dpg.child_window(width=display_w + 20, height=-1):
                    dpg.add_text("Preview", color=(150, 180, 220))
                    dpg.add_image(self.image_texture_id, width=display_w, height=display_h)

                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=5)

                    # Adjustments
                    dpg.add_text("Adjustments", color=(150, 150, 150))
                    dpg.add_slider_float(
                        label="Brightness", default_value=self.brightness,
                        min_value=-0.5, max_value=0.5, format="%.2f", width=200,
                        callback=self._on_brightness_change,
                    )
                    dpg.add_slider_float(
                        label="Contrast", default_value=self.contrast,
                        min_value=0.5, max_value=2.0, format="%.2f", width=200,
                        callback=self._on_contrast_change,
                    )
                    dpg.add_slider_float(
                        label="Gamma", default_value=self.gamma,
                        min_value=0.3, max_value=3.0, format="%.2f", width=200,
                        callback=self._on_gamma_change,
                    )
                    dpg.add_slider_float(
                        label="Clip %", default_value=self.clip_percent,
                        min_value=0.0, max_value=5.0, format="%.2f", width=200,
                        callback=self._on_clip_change,
                    )

                # Right: Palette editor
                with dpg.child_window(width=panel_width, height=-1):
                    dpg.add_text("OKLCH Palette Editor", color=(150, 200, 255))
                    dpg.add_text("Double-click: add/delete stops. Drag handles to reposition.",
                                color=(120, 120, 120))
                    dpg.add_spacer(height=8)

                    # Gradient bar
                    self.gradient_drawlist_id = dpg.add_drawlist(
                        width=self.gradient_width,
                        height=self.gradient_height,
                    )
                    self._update_gradient_drawlist()

                    dpg.add_spacer(height=8)
                    dpg.add_text("Intensity Histogram (post clip/adjust)", color=(150, 150, 150))
                    self.histogram_drawlist_id = dpg.add_drawlist(
                        width=self.hist_width,
                        height=self.hist_height,
                    )
                    self._update_histogram_drawlist()

                    dpg.add_spacer(height=2)
                    dpg.add_button(
                        label="Interp: Rel", width=130,
                        callback=self._on_chroma_interp_toggle, tag="chroma_interp_button",
                    )

                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=8)

                    with dpg.group(horizontal=True):
                        # Left: C x H slice
                        with dpg.group():
                            dpg.add_text("Chroma x Hue")
                            self.slice_drawlist_id = dpg.add_drawlist(
                                width=self.slice_width,
                                height=self.slice_height,
                            )
                            self._update_slice_crosshair()

                        dpg.add_spacer(width=15)

                        # Right: Preview swatch and info
                        with dpg.group():
                            dpg.add_text("Stop")
                            dpg.add_image(self.preview_swatch_id, width=50, height=50)
                            dpg.add_text("", tag="stop_info_text")

                    dpg.add_spacer(height=8)

                    # L/C/H sliders (tight spacing)
                    with dpg.group(tag="slider_block"):
                        dpg.add_image(self.l_gradient_texture_id, width=360, height=14)
                        dpg.add_slider_float(
                            label="Lightness", default_value=0.5,
                            min_value=0.0, max_value=1.0, format="%.3f", width=360,
                            callback=self._on_l_change, tag="l_slider",
                        )

                        with dpg.group(horizontal=True):
                            dpg.add_slider_float(
                                label="Chroma", default_value=0.0,
                                min_value=0.0, max_value=self.c_max_display, format="%.3f", width=360,
                                callback=self._on_c_change, tag="c_slider",
                            )
                            dpg.add_spacer(width=6)
                            dpg.add_checkbox(
                                label="Relative", default_value=self.use_relative_chroma,
                                callback=self._on_chroma_mode_change, tag="chroma_mode_toggle",
                            )

                        dpg.add_slider_float(
                            label="Hue", default_value=0.0,
                            min_value=0.0, max_value=360.0, format="%.1f", width=360,
                            callback=self._on_h_change, tag="h_slider",
                        )

                    if self.slider_block_theme is not None:
                        dpg.bind_item_theme("slider_block", self.slider_block_theme)

                    dpg.add_spacer(height=4)

                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=8)

                    # Save/Load
                    dpg.add_text("Save / Load", color=(150, 150, 150))
                    dpg.add_input_text(label="Name", default_value="", width=200, tag="palette_name_input")
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Save", width=90,
                                      callback=lambda: self._on_save_palette())
                        dpg.add_button(label="New", width=90,
                                      callback=lambda: self._on_new_palette())

                    if self.saved_palettes:
                        dpg.add_spacer(height=5)
                        dpg.add_text("Saved:", color=(120, 120, 120))
                        with dpg.group(horizontal=True):
                            for i, name in enumerate(sorted(self.saved_palettes.keys())[:6]):
                                dpg.add_button(label=name[:12], width=90,
                                              callback=self._on_load_palette, user_data=name)
                                if (i + 1) % 3 == 0:
                                    dpg.end()
                                    dpg.add_group(horizontal=True)

        # Viewport
        vp_width = display_w + panel_width + 60
        vp_height = max(display_h + 150, 750)
        dpg.create_viewport(title="OKLCH Palette Preview", width=vp_width, height=vp_height)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

        self._sync_ui_from_stop()

    def _mark_dirty(self, full: bool = False, hist: bool = False):
        """Mark pending work and process immediately."""
        if full:
            self._update_gradient_texture()
            self._update_slice_texture()
            self._update_l_gradient_texture()
            self._update_preview_swatch()
            self._update_gradient_drawlist()
            self._update_slice_crosshair()
            self._update_image_texture()
            self._sync_ui_from_stop()
        if hist:
            self._update_histogram_drawlist()

    def _process_pending_updates(self, sender=None, app_data=None, user_data=None):
        """Process pending updates immediately (legacy hook)."""
        self._update_gradient_texture()
        self._update_slice_texture()
        self._update_l_gradient_texture()
        self._update_preview_swatch()
        self._update_gradient_drawlist()
        self._update_slice_crosshair()
        self._update_image_texture()
        self._sync_ui_from_stop()

    def run(self):
        self._build_ui()
        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    def pick_default_project() -> Optional[Path]:
        preferred = [
            Path("projects/project.elliptica"),
            Path("project.elliptica"),
        ]
        for candidate in preferred:
            if candidate.exists():
                return candidate

        candidates = []
        projects_dir = Path("projects")
        if projects_dir.exists():
            for path in projects_dir.glob("*.elliptica"):
                if path.with_suffix(".elliptica.cache").exists() or path.with_suffix(".flowcol.cache").exists():
                    candidates.append(path)

        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)

        return None

    if len(sys.argv) == 2:
        project_path = sys.argv[1]
    else:
        default_project = pick_default_project()
        if default_project is None:
            print("Usage: python tools/oklch_palette_preview.py path/to/project.elliptica")
            sys.exit(1)

        project_path = str(default_project)
        print(f"Using default project: {project_path}")

    try:
        app = OklchPalettePreview(project_path)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
