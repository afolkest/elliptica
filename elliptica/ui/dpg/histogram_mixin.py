"""Histogram functionality mixin for PostprocessingPanel.

This mixin provides histogram computation, rendering, and debouncing for
displaying intensity distributions in the postprocessing UI.
"""

import time
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

from elliptica.app.core import resolve_region_postprocess_params

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class HistogramMixin:
    """Mixin providing histogram functionality for PostprocessingPanel.

    This mixin expects the following attributes to be present on the class:
        - app: EllipticaApp instance
        - palette_preview_width: int (width for histogram display)
        - _get_current_region_style_unlocked(): method returning RegionStyle
    """

    # Type hints for attributes expected from the main class
    app: "EllipticaApp"
    palette_preview_width: int

    def _init_histogram_state(self) -> None:
        """Initialize histogram-related instance variables.

        Call this from the main class's __init__ method.
        """
        self.palette_hist_height: int = 60
        self.palette_hist_bins: int = 128
        self.hist_max_samples: int = 1_000_000
        self.hist_target_shape: tuple[int, int] = (512, 512)
        self.global_hist_drawlist_id: Optional[int] = None
        self.region_hist_drawlist_id: Optional[int] = None
        self.global_hist_values: Optional[np.ndarray] = None
        self.region_hist_values: Optional[np.ndarray] = None
        self.hist_pending_update: bool = False
        self.hist_last_update_time: float = 0.0
        self.hist_debounce_delay: float = 0.05  # 50ms throttle

    def _normalize_unit(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            return (arr - lo) / (hi - lo)
        return np.zeros_like(arr, dtype=np.float32)

    def _downsample_histogram_source(self, arr: np.ndarray) -> np.ndarray:
        """Downsample array for efficient histogram computation."""
        target_h, target_w = self.hist_target_shape
        if arr.shape[0] == target_h and arr.shape[1] == target_w:
            return arr.astype(np.float32, copy=False)
        try:
            resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            img = Image.fromarray(arr.astype(np.float32, copy=False), mode="F")
            resized = img.resize((target_w, target_h), resample=resample)
            return np.asarray(resized, dtype=np.float32)
        except Exception:
            max_samples = int(self.hist_max_samples)
            if arr.size <= max_samples:
                return arr.astype(np.float32, copy=False)
            step = max(1, int(np.sqrt(arr.size / max_samples)))
            return arr[::step, ::step].astype(np.float32, copy=False)

    def _compute_histogram_values(
        self,
        source: np.ndarray,
        clip_low: float,
        clip_high: float,
        brightness: float,
        contrast: float,
        gamma: float,
        *,
        cached_percentiles: Optional[tuple[float, float]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Compute histogram for post-processed intensity (clip/contrast/gamma)."""
        if source is None:
            return None

        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.float32)
            if mask_arr.shape != source.shape:
                min_h = min(mask_arr.shape[0], source.shape[0])
                min_w = min(mask_arr.shape[1], source.shape[1])
                if min_h <= 0 or min_w <= 0:
                    return None
                source = source[:min_h, :min_w]
                mask_arr = mask_arr[:min_h, :min_w]
        else:
            mask_arr = None

        sample = self._downsample_histogram_source(source)
        weights = None
        if mask_arr is not None:
            mask_sample = self._downsample_histogram_source(mask_arr)
            if mask_sample.shape != sample.shape:
                min_h = min(mask_sample.shape[0], sample.shape[0])
                min_w = min(mask_sample.shape[1], sample.shape[1])
                if min_h <= 0 or min_w <= 0:
                    return None
                sample = sample[:min_h, :min_w]
                mask_sample = mask_sample[:min_h, :min_w]
            weights = np.clip(mask_sample, 0.0, 1.0)
            if float(weights.max()) <= 0.0:
                return np.zeros(self.palette_hist_bins, dtype=np.float32)

        if clip_low > 0.0 or clip_high > 0.0:
            lower = max(0.0, min(clip_low, 100.0))
            upper = max(0.0, min(100.0 - clip_high, 100.0))
            if upper > lower:
                if cached_percentiles is not None:
                    vmin, vmax = cached_percentiles
                else:
                    vmin = float(np.percentile(sample, lower))
                    vmax = float(np.percentile(sample, upper))
            else:
                vmin = float(sample.min())
                vmax = float(sample.max())
        else:
            vmin = float(sample.min())
            vmax = float(sample.max())

        if vmax > vmin:
            norm = (sample - vmin) / (vmax - vmin)
        else:
            norm = self._normalize_unit(sample)
        norm = np.clip(norm, 0.0, 1.0)

        adjusted = norm
        if contrast != 1.0:
            adjusted = (adjusted - 0.5) * contrast + 0.5
            adjusted = np.clip(adjusted, 0.0, 1.0)
        if brightness != 0.0:
            adjusted = np.clip(adjusted + brightness, 0.0, 1.0)
        if gamma != 1.0:
            adjusted = np.clip(adjusted ** gamma, 0.0, 1.0)

        counts, _ = np.histogram(
            adjusted,
            bins=self.palette_hist_bins,
            range=(0.0, 1.0),
            weights=weights,
        )
        counts = counts.astype(np.float32)
        if counts.max() > 0:
            counts /= counts.max()
        if counts.size >= 3:
            kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
            counts = np.convolve(counts, kernel, mode="same")
            if counts.max() > 0:
                counts /= counts.max()
        return counts

    def _update_histogram_drawlist(self, drawlist_id: Optional[int], values: Optional[np.ndarray]) -> None:
        """Update a histogram drawlist with new values."""
        if dpg is None or drawlist_id is None:
            return

        if not dpg.does_item_exist(drawlist_id):
            return

        dpg.delete_item(drawlist_id, children_only=True)

        if values is None:
            return

        width = float(self.palette_preview_width)
        height = float(self.palette_hist_height)
        padding = 6.0
        bar_top = 0.0
        bar_bottom = height - 2.0
        inner_width = max(1.0, width - 2.0 * padding)
        bin_w = inner_width / float(len(values))

        dpg.draw_rectangle(
            (0, bar_top),
            (width, bar_bottom),
            color=(80, 80, 80, 180),
            thickness=1,
            parent=drawlist_id,
        )

        for i, v in enumerate(values):
            x0 = padding + i * bin_w
            x1 = x0 + bin_w
            y1 = bar_bottom
            y0 = bar_bottom - (v * (height - 4.0))
            dpg.draw_rectangle(
                (x0, y0),
                (x1, y1),
                fill=(170, 175, 190, 180),
                color=(0, 0, 0, 0),
                parent=drawlist_id,
            )

    def _refresh_histogram(self) -> None:
        """Refresh histogram display with current render cache data."""
        if dpg is None:
            return

        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None or cache.result is None:
                self.global_hist_values = None
                self.region_hist_values = None
                self._update_histogram_drawlist(self.global_hist_drawlist_id, None)
                self._update_histogram_drawlist(self.region_hist_drawlist_id, None)
                self.hist_last_update_time = time.time()
                self.hist_pending_update = False
                return

            source = cache.result.array
            clip_low = float(self.app.state.display_settings.clip_low_percent)
            clip_high = float(self.app.state.display_settings.clip_high_percent)
            brightness = float(self.app.state.display_settings.brightness)
            contrast = float(self.app.state.display_settings.contrast)
            gamma = float(self.app.state.display_settings.gamma)

            cached_percentiles = None
            if cache.lic_percentiles is not None:
                cached_clip = cache.lic_percentiles_clip_range
                if cached_clip is not None:
                    cached_low, cached_high = cached_clip
                    if abs(cached_low - clip_low) < 0.01 and abs(cached_high - clip_high) < 0.01:
                        cached_percentiles = cache.lic_percentiles

            region_mask = None
            region_brightness = brightness
            region_contrast = contrast
            region_gamma = gamma
            boundary_idx = -1
            selected = self.app.state.get_selected()
            boundary_selected = selected is not None and selected.id is not None
            if boundary_selected:
                boundary_idx = self.app.state.get_single_selected_idx()
                region_style = self._get_current_region_style_unlocked()
                if region_style is not None:
                    region_brightness, region_contrast, region_gamma = resolve_region_postprocess_params(
                        region_style,
                        self.app.state.display_settings,
                    )
                if boundary_idx >= 0:
                    if self.app.state.selected_region_type == "surface":
                        masks = cache.boundary_masks
                    else:
                        masks = cache.interior_masks
                    if masks is not None and boundary_idx < len(masks):
                        region_mask = masks[boundary_idx]

        self.global_hist_values = self._compute_histogram_values(
            source,
            clip_low,
            clip_high,
            brightness,
            contrast,
            gamma,
            cached_percentiles=cached_percentiles,
        )

        if boundary_idx >= 0:
            self.region_hist_values = self._compute_histogram_values(
                source,
                clip_low,
                clip_high,
                region_brightness,
                region_contrast,
                region_gamma,
                cached_percentiles=cached_percentiles,
                mask=region_mask,
            )
        else:
            self.region_hist_values = self.global_hist_values

        self._update_histogram_drawlist(self.global_hist_drawlist_id, self.global_hist_values)
        self._update_histogram_drawlist(self.region_hist_drawlist_id, self.region_hist_values)
        self.hist_last_update_time = time.time()
        self.hist_pending_update = False

    def _request_histogram_update(self, force: bool = False) -> None:
        """Request a histogram update, respecting debounce timing."""
        if dpg is None:
            return

        now = time.time()
        if force or (now - self.hist_last_update_time) >= self.hist_debounce_delay:
            self._refresh_histogram()
            return

        self.hist_pending_update = True

    def check_histogram_debounce(self) -> None:
        """Check if a pending histogram update should be applied.

        Call this from the main loop to handle debounced updates.
        """
        if not self.hist_pending_update:
            return

        now = time.time()
        if (now - self.hist_last_update_time) >= self.hist_debounce_delay:
            self._refresh_histogram()
