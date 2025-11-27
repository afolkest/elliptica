"""Cache management panel controller for Elliptica UI."""
from elliptica.postprocess.masks import rasterize_conductor_masks
from typing import Optional, TYPE_CHECKING

from elliptica.serialization import compute_project_fingerprint

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class CacheManagementPanel:
    """Controller for cache status display and cache operations."""

    def __init__(self, app: "EllipticaApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main EllipticaApp instance
        """
        self.app = app

        # Widget IDs for cache status display
        self.cache_status_text_id: Optional[int] = None
        self.cache_warning_group_id: Optional[int] = None
        self.mark_clean_button_id: Optional[int] = None
        self.discard_cache_button_id: Optional[int] = None
        self.view_postprocessing_button_id: Optional[int] = None

    def build_cache_status_ui(self, parent) -> None:
        """Build cache status display UI in render controls panel.

        Args:
            parent: Parent widget ID to add cache status widgets to
        """
        if dpg is None:
            return

        dpg.add_text("Render Cache", parent=parent)
        self.cache_status_text_id = dpg.add_text("No cached render", parent=parent)
        with dpg.group(parent=parent) as cache_warning_group:
            self.cache_warning_group_id = cache_warning_group
            dpg.add_text("⚠️  Project modified since render", color=(255, 200, 100))
            with dpg.group(horizontal=True):
                self.mark_clean_button_id = dpg.add_button(
                    label="Mark Clean",
                    callback=self.on_mark_clean_clicked,
                    width=90
                )
                self.discard_cache_button_id = dpg.add_button(
                    label="Discard",
                    callback=self.on_discard_cache_clicked,
                    width=90
                )
        dpg.configure_item(self.cache_warning_group_id, show=False)

    def build_view_postprocessing_button(self, parent) -> None:
        """Build 'View Postprocessing' button in edit controls panel.

        Args:
            parent: Parent widget ID to add button to
        """
        if dpg is None:
            return

        self.view_postprocessing_button_id = dpg.add_button(
            label="View Postprocessing",
            callback=self.on_view_postprocessing_clicked,
            show=False,
            parent=parent
        )

    def update_cache_status_display(self) -> None:
        """Update cache status text and warning visibility."""
        if dpg is None:
            return

        with self.app.state_lock:
            cache = self.app.state.render_cache

            if cache is None:
                # No cache
                if self.cache_status_text_id:
                    dpg.set_value(self.cache_status_text_id, "No cached render")
                if self.cache_warning_group_id:
                    dpg.configure_item(self.cache_warning_group_id, show=False)
                return

            # Cache exists - show resolution
            shape = cache.result.array.shape
            status_text = f"✓ {shape[1]}×{shape[0]} @ {cache.supersample}×"

            # Check if dirty (fingerprint mismatch)
            current_fp = compute_project_fingerprint(self.app.state.project)
            is_dirty = (cache.project_fingerprint != current_fp)

            if is_dirty:
                status_text = f"⚠️  {shape[1]}×{shape[0]} @ {cache.supersample}× (modified)"
                if self.cache_warning_group_id:
                    dpg.configure_item(self.cache_warning_group_id, show=True)
            else:
                if self.cache_warning_group_id:
                    dpg.configure_item(self.cache_warning_group_id, show=False)

            if self.cache_status_text_id:
                dpg.set_value(self.cache_status_text_id, status_text)

    def on_mark_clean_clicked(self, sender=None, app_data=None) -> None:
        """Mark cache as clean (reset fingerprint to current project state)."""
        if dpg is None:
            return

        with self.app.state_lock:
            if self.app.state.render_cache is not None:
                current_fp = compute_project_fingerprint(self.app.state.project)
                self.app.state.render_cache.project_fingerprint = current_fp

        self.update_cache_status_display()
        dpg.set_value("status_text", "Render cache marked as clean")

    def on_discard_cache_clicked(self, sender=None, app_data=None) -> None:
        """Discard cached render."""
        if dpg is None:
            return

        with self.app.state_lock:
            self.app.state.render_cache = None
            self.app.state.view_mode = "edit"

        self.app._update_control_visibility()
        self.app.canvas_renderer.mark_dirty()
        self.update_cache_status_display()
        dpg.set_value("status_text", "Render cache discarded")

    def on_view_postprocessing_clicked(self, sender=None, app_data=None) -> None:
        """Switch to render mode using existing cache (no re-render)."""
        if dpg is None:
            return

        with self.app.state_lock:
            if self.app.state.render_cache is None:
                dpg.set_value("status_text", "No cached render available")
                return
            self.app.state.view_mode = "render"

        self.app._update_control_visibility()
        self.app.file_io.sync_palette_ui()  # Sync palette text when entering render mode
        self.app.display_pipeline.refresh_display()
        dpg.set_value("status_text", "Viewing cached render")

    def rebuild_cache_display_fields(self) -> None:
        """Rebuild masks and percentiles from loaded cache RenderResult.

        This is called after loading a project with a cached render to reconstruct
        the masks and percentiles needed for postprocessing. All kept at full render resolution.
        """
        import numpy as np


        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None or cache.result is None:
                return

            # Load or rebuild masks at full render resolution
            if self.app.state.project.conductors:
                if cache.result.conductor_masks_canvas is not None:
                    # Use pre-computed masks from render
                    cache.conductor_masks = cache.result.conductor_masks_canvas
                    cache.interior_masks = cache.result.interior_masks_canvas
                else:
                    # Fallback: rasterize masks (for compatibility with older cached renders)
                    scale = cache.multiplier * cache.supersample
                    conductor_masks, interior_masks = rasterize_conductor_masks(
                        self.app.state.project.conductors,
                        cache.result.canvas_scaled_shape,
                        cache.result.margin,
                        scale,
                        cache.result.offset_x,
                        cache.result.offset_y,
                    )
                    cache.conductor_masks = conductor_masks
                    cache.interior_masks = interior_masks

            # Recompute LIC percentiles for smear (critical for loaded caches!)
            # ALWAYS compute if missing - don't check smear_enabled, user may enable it later

            if cache.lic_percentiles is None:
                vmin = float(np.percentile(cache.result.array, 0.5))
                vmax = float(np.percentile(cache.result.array, 99.5))
                cache.lic_percentiles = (vmin, vmax)
                cache.lic_percentiles_clip_percent = 0.5

            # Upload to GPU for fast postprocessing (if available)
            try:
                from elliptica.gpu import GPUContext
                if GPUContext.is_available():
                    cache.result_gpu = GPUContext.to_gpu(cache.result.array)
                    cache.ex_gpu = GPUContext.to_gpu(cache.result.ex)
                    cache.ey_gpu = GPUContext.to_gpu(cache.result.ey)
            except Exception as e:
                pass  # Graceful fallback if GPU upload fails
