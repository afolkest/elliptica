"""Cache management panel controller for FlowCol UI."""
from flowcol.render import downsample_lic
from flowcol.postprocess.masks import rasterize_conductor_masks
from scipy.ndimage import zoom
from typing import Optional, TYPE_CHECKING

from flowcol.serialization import compute_project_fingerprint

if TYPE_CHECKING:
    from flowcol.ui.dpg.app import FlowColApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class CacheManagementPanel:
    """Controller for cache status display and cache operations."""

    def __init__(self, app: "FlowColApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main FlowColApp instance
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
        self.app._mark_canvas_dirty()
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
        self.app._mark_canvas_dirty()
        self.app.texture_manager.refresh_render_texture()
        dpg.set_value("status_text", "Viewing cached render")

    def rebuild_cache_display_fields(self) -> None:
        """Rebuild display_array and masks from loaded cache RenderResult.

        This is called after loading a project with a cached render to reconstruct
        the display arrays and masks needed for postprocessing.
        """

        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None or cache.result is None:
                return

            # Recompute display_array
            canvas_w, canvas_h = self.app.state.project.canvas_resolution
            target_shape = (canvas_h, canvas_w)
            display_array = downsample_lic(
                cache.result.array,
                target_shape,
                cache.supersample,
                self.app.state.display_settings.downsample_sigma,
            )
            # Set CPU as primary source (this is CPU-only path)
            cache.set_display_array_cpu(display_array)

            # Use cached masks from RenderResult if available (avoids redundant rasterization)
            if self.app.state.project.conductors:
                if cache.result.conductor_masks_canvas is not None:
                    # Use pre-computed masks from render
                    full_res_conductor_masks = cache.result.conductor_masks_canvas
                    full_res_interior_masks = cache.result.interior_masks_canvas
                else:
                    # Fallback: rasterize masks (for compatibility with older cached renders)
                    scale = cache.multiplier * cache.supersample
                    full_res_conductor_masks, full_res_interior_masks = rasterize_conductor_masks(
                        self.app.state.project.conductors,
                        cache.result.canvas_scaled_shape,
                        cache.result.margin,
                        scale,
                        cache.result.offset_x,
                        cache.result.offset_y,
                    )

                # Store full-resolution masks (for edge blur)
                cache.full_res_conductor_masks = full_res_conductor_masks
                cache.full_res_interior_masks = full_res_interior_masks

                # Downsample masks to match display_array resolution (for region overlays)
                if cache.result.array.shape != display_array.shape:
                    scale_y = display_array.shape[0] / cache.result.array.shape[0]
                    scale_x = display_array.shape[1] / cache.result.array.shape[1]
                    conductor_masks = [
                        zoom(mask, (scale_y, scale_x), order=1) if mask is not None else None
                        for mask in full_res_conductor_masks
                    ]
                    interior_masks = [
                        zoom(mask, (scale_y, scale_x), order=1) if mask is not None else None
                        for mask in full_res_interior_masks
                    ]
                else:
                    # Same resolution - reuse full-res masks
                    conductor_masks = full_res_conductor_masks
                    interior_masks = full_res_interior_masks

                cache.conductor_masks = conductor_masks
                cache.interior_masks = interior_masks
