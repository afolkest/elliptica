"""Render orchestrator controller for background render job management."""

from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import replace
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from flowcol.app.core import RenderCache
from flowcol.pipeline import perform_render
from flowcol.serialization import compute_project_fingerprint

if TYPE_CHECKING:
    from flowcol.ui.dpg.app import FlowColApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class RenderOrchestrator:
    """Controller for background render job management."""

    def __init__(self, app: "FlowColApp"):
        """Initialize orchestrator with reference to main app.

        Args:
            app: The main FlowColApp instance
        """
        self.app = app

        # Executor for background render jobs
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

        # Current render job state
        self.render_future: Optional[Future] = None
        self.render_error: Optional[str] = None

    def start_job(self) -> None:
        """Start a background render job with current settings.

        If a render is already in progress, this method returns immediately.
        The render runs in a background thread and updates app state when complete.
        """
        if self.render_future is not None and not self.render_future.done():
            return

        self.render_error = None
        if dpg is not None:
            dpg.set_value("status_text", "Rendering...")

        def job() -> bool:
            """Background render job that runs in executor thread."""
            # Import the snapshot helper from app module
            from flowcol.ui.dpg.app import _snapshot_project

            # Snapshot settings and project for thread-safe rendering
            with self.app.state_lock:
                settings_snapshot = replace(self.app.state.render_settings)
                project_snapshot = _snapshot_project(self.app.state.project)

            # Perform the expensive render operation
            result = perform_render(
                project_snapshot,
                settings_snapshot.multiplier,
                settings_snapshot.supersample,
                settings_snapshot.num_passes,
                settings_snapshot.margin,
                settings_snapshot.noise_seed,
                settings_snapshot.noise_sigma,
                project_snapshot.streamlength_factor,
            )
            if result is None:
                return False

            # Apply initial postprocessing to create display array
            from flowcol.render import downsample_lic

            with self.app.state_lock:
                postprocess = self.app.state.display_settings
                # canvas_resolution is (width, height), but downsample_lic expects (height, width)
                canvas_w, canvas_h = project_snapshot.canvas_resolution
                target_shape = (canvas_h, canvas_w)
                display_array = downsample_lic(
                    result.array,
                    target_shape,
                    settings_snapshot.supersample,
                    postprocess.downsample_sigma,
                )

            # Use cached masks from RenderResult (avoids redundant rasterization)
            full_res_conductor_masks = result.conductor_masks_canvas
            full_res_interior_masks = result.interior_masks_canvas
            conductor_masks = None
            interior_masks = None
            if project_snapshot.conductors and full_res_conductor_masks is not None:
                from scipy.ndimage import zoom

                # Downsample masks to match display_array resolution
                if result.array.shape != display_array.shape:
                    scale_y = display_array.shape[0] / result.array.shape[0]
                    scale_x = display_array.shape[1] / result.array.shape[1]
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

            # Create render cache
            cache = RenderCache(
                result=result,
                multiplier=settings_snapshot.multiplier,
                supersample=settings_snapshot.supersample,
                display_array=display_array,
                base_rgb=None,  # Will be built on-demand
                conductor_masks=conductor_masks,
                interior_masks=interior_masks,
                full_res_conductor_masks=full_res_conductor_masks,
                full_res_interior_masks=full_res_interior_masks,
            )

            # Set fingerprint for cache staleness detection
            cache.project_fingerprint = compute_project_fingerprint(project_snapshot)

            # Upload render result to GPU for fast postprocessing
            try:
                from flowcol.gpu import GPUContext
                print(f"DEBUG render: GPU available? {GPUContext.is_available()}")
                print(f"DEBUG render: result has ex? {hasattr(result, 'ex')}")
                print(f"DEBUG render: result has ey? {hasattr(result, 'ey')}")
                if GPUContext.is_available():
                    cache.result_gpu = GPUContext.to_gpu(result.array)
                    cache.ex_gpu = GPUContext.to_gpu(result.ex)
                    cache.ey_gpu = GPUContext.to_gpu(result.ey)
                    print(f"DEBUG render: result_gpu = {cache.result_gpu is not None}")
                    print(f"DEBUG render: ex_gpu = {cache.ex_gpu is not None}")
                    print(f"DEBUG render: ey_gpu = {cache.ey_gpu is not None}")
            except Exception as e:
                print(f"DEBUG render: EXCEPTION during GPU upload: {e}")
                pass  # Graceful fallback if GPU upload fails

            # Update app state with completed render
            with self.app.state_lock:
                self.app.state.render_cache = cache
                self.app.state.field_dirty = False
                self.app.state.render_dirty = False
                self.app.state.view_mode = "render"

            # Auto-save high-res renders (>2k on any dimension)
            render_h, render_w = result.array.shape
            if render_w >= 2000 or render_h >= 2000:
                from PIL import Image
                from datetime import datetime

                output_dir = Path.cwd() / "output_raw"
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if settings_snapshot.supersample > 1.0:
                    # Save supersampled raw LIC
                    output_path_super = output_dir / f"render_{render_w}x{render_h}_supersampled_{timestamp}.png"
                    img_data_super = (np.clip(result.array, 0, 1) * 255).astype(np.uint8)
                    pil_img_super = Image.fromarray(img_data_super, mode='L')
                    pil_img_super.save(output_path_super)
                    print(f"Auto-saved supersampled render to: {output_path_super.name}")

                    # Downsample and save final raw LIC
                    canvas_w, canvas_h = project_snapshot.canvas_resolution
                    output_canvas_w = int(round(canvas_w * settings_snapshot.multiplier))
                    output_canvas_h = int(round(canvas_h * settings_snapshot.multiplier))
                    output_shape = (output_canvas_h, output_canvas_w)

                    downsampled_lic = downsample_lic(
                        result.array,
                        output_shape,
                        settings_snapshot.supersample,
                        0.6,  # Default sigma
                    )

                    output_h, output_w = downsampled_lic.shape
                    output_path_final = output_dir / f"render_{output_w}x{output_h}_final_{timestamp}.png"
                    img_data_final = (np.clip(downsampled_lic, 0, 1) * 255).astype(np.uint8)
                    pil_img_final = Image.fromarray(img_data_final, mode='L')
                    pil_img_final.save(output_path_final)
                    print(f"Auto-saved final render to: {output_path_final.name}")
                else:
                    # Single save
                    output_path = output_dir / f"render_{render_w}x{render_h}_{timestamp}.png"
                    img_data = (np.clip(result.array, 0, 1) * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_data, mode='L')
                    pil_img.save(output_path)
                    print(f"Auto-saved high-res render to: {output_path.name}")

            return True

        # Submit job to executor
        self.render_future = self.executor.submit(job)

    def poll(self) -> None:
        """Poll the render future and handle completion.

        Should be called every frame from the main event loop.
        When the render completes, updates the UI and triggers callbacks.
        """
        if self.render_future is None:
            return
        if not self.render_future.done():
            return

        # Render completed - get result
        success = False
        try:
            success = bool(self.render_future.result())
        except Exception as exc:  # pragma: no cover - unexpected
            success = False
            self.render_error = str(exc)

        self.render_future = None

        if success:
            # Render succeeded - update UI
            self.app._mark_canvas_dirty()
            self.app.texture_manager.refresh_render_texture()
            self.app.drag_active = False
            with self.app.state_lock:
                self.app.state.view_mode = "render"
            self.app._update_control_visibility()
            self.app.cache_panel.update_cache_status_display()

            # Auto-save cache if project has been saved before
            if self.app.file_io.current_project_path is not None:
                self.app.file_io.auto_save_cache()
            else:
                if dpg is not None:
                    dpg.set_value("status_text", "Render complete. (Save project to preserve cache)")
        else:
            # Render failed
            msg = self.render_error or "Render failed (possibly due to excessive resolution)."
            if dpg is not None:
                dpg.set_value("status_text", msg)

    def shutdown(self, wait: bool = False) -> None:
        """Shutdown the executor thread pool.

        Args:
            wait: If True, wait for pending jobs to complete before shutting down
        """
        self.executor.shutdown(wait=wait)
