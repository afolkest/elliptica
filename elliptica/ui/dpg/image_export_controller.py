"""Image export controller for Elliptica UI.

Handles saving rendered images to disk with support for:
- Quick save (current display as-is)
- High-resolution export with re-rendering
- Postprocessing pipeline at export resolution
"""

from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PIL import Image
from datetime import datetime

from elliptica.postprocess.masks import rasterize_boundary_masks
from elliptica.render import downsample_lic

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp, _snapshot_project
    from elliptica.types import Project

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class ImageExportController:
    """Controller for image export operations."""

    def __init__(self, app: "EllipticaApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main EllipticaApp instance
        """
        self.app = app
        self.export_modal_id: Optional[int] = None
        self.export_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self.export_future: Optional[Future] = None
        # Pending export settings (stored when file dialog opens)
        self._pending_export_multiplier: float = 1.0
        self._pending_export_solve_scale: float = 1.0
        self._pending_quick_save_path: Optional[Path] = None
        self._pending_export_path: Optional[Path] = None
        self._overwrite_confirm_id: Optional[int] = None

    def quick_save(self, sender=None, app_data=None) -> None:
        """Save the current display buffer directly to disk.

        This is a fast operation - no re-rendering, just saves what's currently shown.
        """
        if dpg is None:
            return

        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None or cache.result is None:
                dpg.set_value("status_text", "No render to save.")
                return

        # Get the current display texture data from texture manager
        texture_manager = self.app.display_pipeline.texture_manager
        if texture_manager.render_texture_id is None:
            dpg.set_value("status_text", "No display to save.")
            return

        # Create file dialog if needed
        if not dpg.does_item_exist("quick_save_file_dialog"):
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=self._on_quick_save_file_selected,
                tag="quick_save_file_dialog",
                width=700,
                height=400,
                default_filename="elliptica_quicksave.png",
            ):
                dpg.add_file_extension(".png", color=(0, 255, 0, 255))
                dpg.add_file_extension(".jpg", color=(0, 255, 255, 255))

        # Set default path to outputs directory
        output_dir = Path.cwd() / "outputs"
        output_dir.mkdir(exist_ok=True)
        dpg.configure_item("quick_save_file_dialog", default_path=str(output_dir))
        dpg.configure_item("quick_save_file_dialog", show=True)

    def _on_quick_save_file_selected(self, sender=None, app_data=None) -> None:
        """Handle file selection for quick save."""
        if dpg is None:
            return

        if app_data is None or 'file_path_name' not in app_data:
            return

        file_path = Path(app_data['file_path_name'])

        # Check if file exists and confirm overwrite
        if file_path.exists():
            self._pending_quick_save_path = file_path
            self._show_overwrite_confirm("quick_save")
            return

        self._do_quick_save(file_path)

    def _do_quick_save(self, file_path: Path) -> None:
        """Actually perform the quick save."""
        # Get the current display texture data from texture manager
        texture_manager = self.app.display_pipeline.texture_manager
        texture_data = dpg.get_value(texture_manager.render_texture_id)
        if texture_data is None:
            dpg.set_value("status_text", "Failed to get texture data.")
            return

        # Convert flat RGBA float data back to image
        width, height = texture_manager.render_texture_size
        rgba_flat = np.array(texture_data, dtype=np.float32)
        rgba = (rgba_flat.reshape(height, width, 4) * 255).astype(np.uint8)
        rgb = rgba[:, :, :3]  # Drop alpha

        # Save
        pil_img = Image.fromarray(rgb, mode='RGB')
        pil_img.save(file_path)
        dpg.set_value("status_text", f"Saved {file_path.name}")

    def open_export_dialog(self, sender=None, app_data=None) -> None:
        """Open the export settings dialog."""
        if dpg is None:
            return

        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None or cache.result is None:
                dpg.set_value("status_text", "No render to export.")
                return

        self._ensure_export_modal()

        # Get current render info for display
        with self.app.state_lock:
            cache = self.app.state.render_cache
            current_h, current_w = cache.result.array.shape
            current_multiplier = cache.multiplier
            current_solve_scale = self.app.state.render_settings.solve_scale

        # Update info text
        dpg.set_value("export_current_res_text", f"Current: {current_w} x {current_h}")

        # Set defaults
        dpg.set_value("export_multiplier_input", current_multiplier)
        dpg.set_value("export_solve_scale_slider", current_solve_scale)

        # Update preview
        self._update_export_preview()

        # Center and show modal
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        modal_width = 400
        modal_height = 300
        dpg.configure_item(
            self.export_modal_id,
            pos=((viewport_width - modal_width) // 2, (viewport_height - modal_height) // 2),
            show=True
        )

    def _ensure_export_modal(self) -> None:
        """Create the export modal dialog if it doesn't exist."""
        if dpg is None or self.export_modal_id is not None:
            return

        with dpg.window(
            label="Export Image",
            modal=True,
            show=False,
            tag="export_modal",
            no_resize=True,
            no_collapse=True,
            width=400,
            height=300,
        ) as modal:
            self.export_modal_id = modal

            dpg.add_text("Export Settings", color=(200, 200, 255))
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Current resolution info
            dpg.add_text("", tag="export_current_res_text", color=(150, 150, 150))
            dpg.add_spacer(height=10)

            # Resolution multiplier
            with dpg.group(horizontal=True):
                dpg.add_text("Resolution multiplier:")
                dpg.add_input_float(
                    default_value=1.0,
                    min_value=0.25,
                    max_value=16.0,
                    step=0.25,
                    width=100,
                    tag="export_multiplier_input",
                    callback=self._update_export_preview,
                )

            # Preview resolution
            dpg.add_text("", tag="export_preview_res_text", color=(100, 200, 100))
            dpg.add_spacer(height=10)

            # PDE solve scale
            with dpg.group(horizontal=True):
                dpg.add_text("PDE solve scale:")
                dpg.add_slider_float(
                    default_value=1.0,
                    min_value=0.1,
                    max_value=1.0,
                    format="%.2f",
                    width=100,
                    tag="export_solve_scale_slider",
                )
            dpg.add_text("1 = best quality, lower = faster", color=(150, 150, 150))

            dpg.add_spacer(height=20)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Export...",
                    width=100,
                    callback=self._on_export_clicked,
                )
                dpg.add_button(
                    label="Cancel",
                    width=100,
                    callback=lambda: dpg.configure_item("export_modal", show=False),
                )

    def _update_export_preview(self, sender=None, app_data=None) -> None:
        """Update the export preview resolution text."""
        if dpg is None:
            return

        multiplier = dpg.get_value("export_multiplier_input")

        with self.app.state_lock:
            canvas_w, canvas_h = self.app.state.project.canvas_resolution

        export_w = int(round(canvas_w * multiplier))
        export_h = int(round(canvas_h * multiplier))

        dpg.set_value("export_preview_res_text", f"Export size: {export_w} x {export_h}")

    def _on_export_clicked(self, sender=None, app_data=None) -> None:
        """Handle export button click - open file dialog."""
        if dpg is None:
            return

        # Get and store export settings
        self._pending_export_multiplier = dpg.get_value("export_multiplier_input")
        self._pending_export_solve_scale = float(dpg.get_value("export_solve_scale_slider"))

        with self.app.state_lock:
            canvas_w, canvas_h = self.app.state.project.canvas_resolution

        export_w = int(round(canvas_w * self._pending_export_multiplier))
        export_h = int(round(canvas_h * self._pending_export_multiplier))

        # Hide the export settings modal before showing file dialog
        dpg.configure_item("export_modal", show=False)

        # Create file dialog if needed
        if not dpg.does_item_exist("export_file_dialog"):
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=self._on_export_file_selected,
                tag="export_file_dialog",
                width=700,
                height=400,
            ):
                dpg.add_file_extension(".png", color=(0, 255, 0, 255))
                dpg.add_file_extension(".tiff", color=(255, 255, 0, 255))

        # Set default filename and path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"elliptica_{export_w}x{export_h}_{timestamp}.png"

        output_dir = Path.cwd() / "outputs"
        output_dir.mkdir(exist_ok=True)

        dpg.configure_item("export_file_dialog", default_path=str(output_dir))
        dpg.configure_item("export_file_dialog", default_filename=default_filename)
        dpg.configure_item("export_file_dialog", show=True)

    def _on_export_file_selected(self, sender=None, app_data=None) -> None:
        """Handle file selection for export."""
        if dpg is None:
            return

        if app_data is None or 'file_path_name' not in app_data:
            return

        file_path = Path(app_data['file_path_name'])

        # Check if file exists and confirm overwrite
        if file_path.exists():
            self._pending_export_path = file_path
            self._show_overwrite_confirm("export")
            return

        # Start export in background with stored settings
        self._start_export(file_path, self._pending_export_multiplier, self._pending_export_solve_scale)

    def _show_overwrite_confirm(self, mode: str) -> None:
        """Show overwrite confirmation dialog."""
        if dpg is None:
            return

        # Create dialog if needed
        if self._overwrite_confirm_id is None:
            with dpg.window(
                label="Confirm Overwrite",
                modal=True,
                show=False,
                tag="overwrite_confirm_modal",
                no_resize=True,
                no_collapse=True,
                width=350,
                height=120,
            ) as modal:
                self._overwrite_confirm_id = modal
                dpg.add_text("", tag="overwrite_confirm_text")
                dpg.add_spacer(height=10)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Overwrite", width=100, callback=self._on_overwrite_confirmed)
                    dpg.add_button(label="Cancel", width=100, callback=self._on_overwrite_cancelled)

        # Set the message
        if mode == "quick_save":
            filename = self._pending_quick_save_path.name if self._pending_quick_save_path else "file"
        else:
            filename = self._pending_export_path.name if self._pending_export_path else "file"

        dpg.set_value("overwrite_confirm_text", f"'{filename}' already exists. Overwrite?")
        dpg.set_item_user_data("overwrite_confirm_modal", mode)

        # Center and show
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dpg.configure_item(
            self._overwrite_confirm_id,
            pos=((viewport_width - 350) // 2, (viewport_height - 120) // 2),
            show=True
        )

    def _on_overwrite_confirmed(self, sender=None, app_data=None) -> None:
        """Handle overwrite confirmation."""
        if dpg is None:
            return

        mode = dpg.get_item_user_data("overwrite_confirm_modal")
        dpg.configure_item("overwrite_confirm_modal", show=False)

        if mode == "quick_save" and self._pending_quick_save_path:
            self._do_quick_save(self._pending_quick_save_path)
            self._pending_quick_save_path = None
        elif mode == "export" and self._pending_export_path:
            self._start_export(self._pending_export_path, self._pending_export_multiplier, self._pending_export_solve_scale)
            self._pending_export_path = None

    def _on_overwrite_cancelled(self, sender=None, app_data=None) -> None:
        """Handle overwrite cancellation."""
        if dpg is None:
            return

        dpg.configure_item("overwrite_confirm_modal", show=False)
        self._pending_quick_save_path = None
        self._pending_export_path = None

    def _start_export(self, file_path: Path, multiplier: float, solve_scale: float) -> None:
        """Start the export process in a background thread."""
        if dpg is None:
            return

        dpg.set_value("status_text", f"Exporting at {multiplier}x resolution...")

        # Snapshot all settings for thread-safe export
        with self.app.state_lock:
            from elliptica.ui.dpg.app import _snapshot_project
            project = _snapshot_project(self.app.state.project)
            settings = replace(self.app.state.display_settings)
            render_settings = replace(self.app.state.render_settings)
            boundary_color_settings = {k: v for k, v in self.app.state.boundary_color_settings.items()}
            color_config = self.app.state.color_config

        def export_job():
            """Background export job."""
            from elliptica.pipeline import perform_render
            from elliptica.gpu.postprocess import apply_full_postprocess_hybrid

            # Perform render at export resolution
            result = perform_render(
                project,
                multiplier,
                render_settings.supersample,
                render_settings.num_passes,
                render_settings.margin,
                render_settings.noise_seed,
                render_settings.noise_sigma,
                project.streamlength_factor,
                render_settings.use_mask,
                render_settings.edge_gain_strength,
                render_settings.edge_gain_power,
                render_settings.domain_edge_gain_strength,
                render_settings.domain_edge_gain_power,
                solve_scale,
            )

            if result is None:
                return ('error', 'Render failed')

            # Rasterize masks at export resolution
            boundary_masks = None
            interior_masks = None
            if project.boundary_objects:
                boundary_masks, interior_masks = rasterize_boundary_masks(
                    project.boundary_objects,
                    result.array.shape,
                    result.margin,
                    multiplier * render_settings.supersample,
                    result.offset_x,
                    result.offset_y,
                )

            # Upload solution and field components to GPU if available
            solution_gpu = None
            ex_tensor = None
            ey_tensor = None
            try:
                from elliptica.gpu import GPUContext
                if GPUContext.is_available():
                    if result.solution:
                        solution_gpu = {}
                        for name, array in result.solution.items():
                            if isinstance(array, np.ndarray) and array.ndim == 2:
                                solution_gpu[name] = GPUContext.to_gpu(array)
                    if result.ex is not None:
                        ex_tensor = GPUContext.to_gpu(result.ex)
                    if result.ey is not None:
                        ey_tensor = GPUContext.to_gpu(result.ey)
            except Exception:
                pass

            # Apply postprocessing
            canvas_w, canvas_h = project.canvas_resolution
            final_rgb, _ = apply_full_postprocess_hybrid(
                scalar_array=result.array,
                boundary_masks=boundary_masks,
                interior_masks=interior_masks,
                boundary_color_settings=boundary_color_settings,
                boundaries=project.boundary_objects,
                render_shape=result.array.shape,
                canvas_resolution=(canvas_w, canvas_h),
                clip_low_percent=settings.clip_low_percent,
                clip_high_percent=settings.clip_high_percent,
                brightness=settings.brightness,
                contrast=settings.contrast,
                gamma=settings.gamma,
                color_enabled=settings.color_enabled,
                palette=settings.palette,
                lic_percentiles=None,
                use_gpu=True,
                color_config=color_config,
                lightness_expr=settings.lightness_expr,
                solution_gpu=solution_gpu,
                ex_tensor=ex_tensor,
                ey_tensor=ey_tensor,
                saturation=settings.saturation,
            )

            # Handle supersampling if enabled
            if render_settings.supersample > 1.0:
                # Downsample to final resolution
                output_h = int(round(canvas_h * multiplier))
                output_w = int(round(canvas_w * multiplier))
                output_shape = (output_h, output_w)

                # Downsample each channel
                from scipy.ndimage import zoom
                scale_factor = 1.0 / render_settings.supersample
                final_rgb_downsampled = zoom(
                    final_rgb,
                    (scale_factor, scale_factor, 1),
                    order=1,
                )
                final_rgb = np.clip(final_rgb_downsampled, 0, 255).astype(np.uint8)

            # Save image
            pil_img = Image.fromarray(final_rgb, mode='RGB')
            pil_img.save(file_path)

            return ('success', file_path.name)

        def on_complete(future):
            """Handle export completion on main thread."""
            try:
                result = future.result()
                if result[0] == 'success':
                    if dpg is not None:
                        dpg.set_value("status_text", f"Exported {result[1]}")
                else:
                    if dpg is not None:
                        dpg.set_value("status_text", f"Export failed: {result[1]}")
            except Exception as e:
                if dpg is not None:
                    dpg.set_value("status_text", f"Export error: {e}")

        self.export_future = self.export_executor.submit(export_job)
        self.export_future.add_done_callback(on_complete)

    def shutdown(self) -> None:
        """Shutdown the export executor."""
        self.export_executor.shutdown(wait=False)

    def export_image(self, sender=None, app_data=None) -> None:
        """Save the final rendered image to disk.

        If supersampled: saves two versions (_supersampled and _final).
        If not supersampled: saves one version.
        """
        if dpg is None:
            return

        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None or cache.result is None:
                dpg.set_value("status_text", "No render to save.")
                return

            # Snapshot everything for thread-safe export
            from elliptica.ui.dpg.app import _snapshot_project
            result = cache.result
            project = _snapshot_project(self.app.state.project)
            settings = replace(self.app.state.display_settings)
            boundary_color_settings = {k: v for k, v in self.app.state.boundary_color_settings.items()}
            color_config = self.app.state.color_config  # Expression-based coloring (if set)
            multiplier = cache.multiplier
            supersample = cache.supersample
            # Snapshot cached masks (these are correct - don't regenerate!)
            cached_boundary_masks = [m.copy() if m is not None else None for m in cache.boundary_masks] if cache.boundary_masks else None
            cached_interior_masks = [m.copy() if m is not None else None for m in cache.interior_masks] if cache.interior_masks else None

        canvas_w, canvas_h = project.canvas_resolution
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path.cwd() / "outputs"
        output_dir.mkdir(exist_ok=True)
        margin_physical = result.margin

        # Use render result directly
        lic_array = result.array

        if supersample > 1.0:
            # Save supersampled version at render resolution
            render_scale = multiplier * supersample
            final_rgb_super = self._apply_postprocessing_at_resolution(
                lic_array,
                project,
                settings,
                boundary_color_settings,
                (canvas_w, canvas_h),
                margin_physical,
                render_scale,
                result.offset_x,
                result.offset_y,
                cached_boundary_masks,
                cached_interior_masks,
                color_config,
                result.solution,
            )

            h_super, w_super = final_rgb_super.shape[:2]
            output_path_super = output_dir / f"elliptica_{w_super}x{h_super}_supersampled_{timestamp}.png"
            pil_img_super = Image.fromarray(final_rgb_super, mode='RGB')
            pil_img_super.save(output_path_super)

            # Downsample LIC to output resolution
            output_canvas_w = int(round(canvas_w * multiplier))
            output_canvas_h = int(round(canvas_h * multiplier))
            output_shape = (output_canvas_h, output_canvas_w)

            downsampled_lic = downsample_lic(
                lic_array,
                output_shape,
                supersample,
                settings.downsample_sigma,
            )

            # Save final version at output resolution
            # Scale offsets down by supersample factor
            output_scale = multiplier
            output_offset_x = int(round(result.offset_x / supersample))
            output_offset_y = int(round(result.offset_y / supersample))

            # For downsampled version, regenerate masks at output resolution
            # Note: solution dict not passed here as array shapes won't match downsampled LIC
            final_rgb_output = self._apply_postprocessing_at_resolution(
                downsampled_lic,
                project,
                settings,
                boundary_color_settings,
                (canvas_w, canvas_h),
                margin_physical,
                output_scale,
                output_offset_x,
                output_offset_y,
                None,  # Regenerate masks for downsampled resolution
                None,
                color_config,
                None,  # Solution not available at downsampled resolution
            )

            h_output, w_output = final_rgb_output.shape[:2]
            output_path_final = output_dir / f"elliptica_{w_output}x{h_output}_final_{timestamp}.png"
            pil_img_output = Image.fromarray(final_rgb_output, mode='RGB')
            pil_img_output.save(output_path_final)

            dpg.set_value("status_text", f"Saved {output_path_super.name} and {output_path_final.name}")
        else:
            # No supersampling: save single version
            render_scale = multiplier
            final_rgb = self._apply_postprocessing_at_resolution(
                lic_array,
                project,
                settings,
                boundary_color_settings,
                (canvas_w, canvas_h),
                margin_physical,
                render_scale,
                result.offset_x,
                result.offset_y,
                cached_boundary_masks,
                cached_interior_masks,
                color_config,
                result.solution,
            )

            h, w = final_rgb.shape[:2]
            output_path = output_dir / f"elliptica_{w}x{h}_{timestamp}.png"
            pil_img = Image.fromarray(final_rgb, mode='RGB')
            pil_img.save(output_path)

            dpg.set_value("status_text", f"Saved {output_path.name}")

    def _apply_postprocessing_at_resolution(
        self,
        lic_array: np.ndarray,
        project: "Project",
        settings,
        boundary_color_settings: dict,
        canvas_resolution: tuple[int, int],
        margin_physical: float,
        scale: float,
        offset_x: int,
        offset_y: int,
        cached_boundary_masks: list = None,
        cached_interior_masks: list = None,
        color_config=None,
        solution: dict = None,
    ) -> np.ndarray:
        """Apply full post-processing pipeline to LIC array at any resolution.

        Args:
            lic_array: Grayscale LIC array
            project: Project snapshot
            settings: Display settings snapshot
            boundary_color_settings: Boundary color settings
            canvas_resolution: Canvas resolution (width, height)
            margin_physical: Physical margin in canvas units
            scale: Pixels per canvas unit (multiplier or multiplier*supersample)
            offset_x: Crop offset X from render result
            offset_y: Crop offset Y from render result

        Returns:
            Final RGB array with all post-processing applied
        """
        # Use cached masks if available (avoids regeneration errors), otherwise generate at this resolution
        if cached_boundary_masks is not None and cached_interior_masks is not None:
            boundary_masks = cached_boundary_masks
            interior_masks = cached_interior_masks
        else:
            boundary_masks = None
            interior_masks = None
            if project.boundary_objects:
                boundary_masks, interior_masks = rasterize_boundary_masks(
                    project.boundary_objects,
                    lic_array.shape,
                    margin_physical,
                    scale,
                    offset_x,
                    offset_y,
                )

        # Compute percentiles for smear (if needed at export resolution)
        lic_percentiles = None
        if any(b.smear_enabled for b in project.boundary_objects):
            clip_low = float(settings.clip_low_percent)
            clip_high = float(settings.clip_high_percent)
            if clip_low > 0.0 or clip_high > 0.0:
                lower = max(0.0, min(clip_low, 100.0))
                upper = max(0.0, min(100.0 - clip_high, 100.0))
                if upper > lower:
                    vmin = float(np.percentile(lic_array, lower))
                    vmax = float(np.percentile(lic_array, upper))
                else:
                    vmin = float(np.min(lic_array))
                    vmax = float(np.max(lic_array))
            else:
                vmin = float(np.min(lic_array))
                vmax = float(np.max(lic_array))
            lic_percentiles = (vmin, vmax)

        # Use unified GPU postprocessing pipeline (or CPU fallback)
        from elliptica.gpu.postprocess import apply_full_postprocess_hybrid

        # Upload solution fields to GPU if available
        solution_gpu = None
        if solution:
            try:
                from elliptica.gpu import GPUContext
                import numpy as np
                if GPUContext.is_available():
                    solution_gpu = {}
                    for name, array in solution.items():
                        if isinstance(array, np.ndarray) and array.ndim == 2:
                            solution_gpu[name] = GPUContext.to_gpu(array)
            except Exception:
                pass  # Graceful fallback if GPU upload fails

        final_rgb, _ = apply_full_postprocess_hybrid(
            scalar_array=lic_array,
            boundary_masks=boundary_masks,
            interior_masks=interior_masks,
            boundary_color_settings=boundary_color_settings,
            boundaries=project.boundary_objects,
            render_shape=lic_array.shape,
            canvas_resolution=canvas_resolution,
            clip_low_percent=settings.clip_low_percent,
            clip_high_percent=settings.clip_high_percent,
            brightness=settings.brightness,
            contrast=settings.contrast,
            gamma=settings.gamma,
            color_enabled=settings.color_enabled,
            palette=settings.palette,
            lic_percentiles=lic_percentiles,
            use_gpu=True,  # GPU acceleration for faster exports
            scalar_tensor=None,  # Will upload on-demand
            color_config=color_config,  # Expression-based coloring (if set)
            lightness_expr=settings.lightness_expr,
            solution_gpu=solution_gpu,
            saturation=settings.saturation,
        )

        return final_rgb
