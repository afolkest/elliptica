"""Render orchestrator controller for background render job management."""

from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import replace
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from elliptica.app.core import RenderCache
from elliptica.pipeline import perform_render
from elliptica.render import downsample_lic
from elliptica.serialization import compute_project_fingerprint

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class RenderOrchestrator:
    """Controller for background render job management."""

    def __init__(self, app: "EllipticaApp"):
        """Initialize orchestrator with reference to main app.

        Args:
            app: The main EllipticaApp instance
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
            from elliptica.ui.dpg.app import _snapshot_project

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
                settings_snapshot.use_mask,
                settings_snapshot.edge_gain_strength,
                settings_snapshot.edge_gain_power,
                settings_snapshot.solve_scale,
            )
            if result is None:
                return False

            # Keep masks at full render resolution (from RenderResult)
            # No downsampling - DearPyGUI will handle display scaling
            conductor_masks = result.conductor_masks_canvas
            interior_masks = result.interior_masks_canvas

            # Precompute LIC percentiles (used for both smear normalization and postprocessing)
            # ALWAYS compute during render to avoid 6s delays later during postprocessing!
            vmin = float(np.percentile(result.array, 0.5))
            vmax = float(np.percentile(result.array, 99.5))
            lic_percentiles = (vmin, vmax)

            # Create render cache (everything at full resolution)
            cache = RenderCache(
                result=result,
                multiplier=settings_snapshot.multiplier,
                supersample=settings_snapshot.supersample,
                base_rgb=None,  # Will be built on-demand during postprocessing
                conductor_masks=conductor_masks,
                interior_masks=interior_masks,
                lic_percentiles=lic_percentiles,
            )

            # Set fingerprint for cache staleness detection
            cache.project_fingerprint = compute_project_fingerprint(project_snapshot)

            # Upload render result to GPU for fast postprocessing
            try:
                from elliptica.gpu import GPUContext
                if GPUContext.is_available():
                    cache.result_gpu = GPUContext.to_gpu(result.array)
                    # Upload field components (may be None for some PDE types)
                    if result.ex is not None:
                        cache.ex_gpu = GPUContext.to_gpu(result.ex)
                    if result.ey is not None:
                        cache.ey_gpu = GPUContext.to_gpu(result.ey)

                    # Upload PDE solution fields to GPU (phi, etc.)
                    if result.solution:
                        cache.solution_gpu = {}
                        cache.solution_gpu_resized = None  # Clear resized cache
                        cache.solution_gpu_lic_shape = None
                        for name, array in result.solution.items():
                            if isinstance(array, np.ndarray) and array.ndim == 2:
                                cache.solution_gpu[name] = GPUContext.to_gpu(array)

                    # Upload conductor masks to GPU (avoids repeated CPU→GPU transfers on every display update)
                    if conductor_masks is not None:
                        cache.conductor_masks_gpu = []
                        for mask in conductor_masks:
                            if mask is not None:
                                cache.conductor_masks_gpu.append(GPUContext.to_gpu(mask))
                            else:
                                cache.conductor_masks_gpu.append(None)

                    if interior_masks is not None:
                        cache.interior_masks_gpu = []
                        for mask in interior_masks:
                            if mask is not None:
                                cache.interior_masks_gpu.append(GPUContext.to_gpu(mask))
                            else:
                                cache.interior_masks_gpu.append(None)
            except Exception as e:
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
            self.app.display_pipeline.refresh_display()
            self.app.canvas_controller.drag_active = False
            with self.app.state_lock:
                self.app.state.view_mode = "render"
                self.app.state.selected_idx = -1  # Clear selection when entering render mode
            self.app._update_control_visibility()
            self.app.postprocess_panel.update_context_ui()  # Update UI for no selection
            self.app.file_io.sync_palette_ui()  # Sync palette text when entering render mode
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
