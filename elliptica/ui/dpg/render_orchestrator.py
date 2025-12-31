"""Render orchestrator controller for background render job management."""

from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import replace
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from elliptica.app.core import RenderCache
from elliptica import defaults
from elliptica.pipeline import perform_render
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
            boundary_masks = result.boundary_masks_canvas
            interior_masks = result.interior_masks_canvas

            # Precompute LIC percentiles (used for both smear normalization and postprocessing)
            # ALWAYS compute during render to avoid 6s delays later during postprocessing!
            vmin = float(np.percentile(result.array, defaults.DEFAULT_CLIP_LOW_PERCENT))
            vmax = float(np.percentile(result.array, 100.0 - defaults.DEFAULT_CLIP_HIGH_PERCENT))
            lic_percentiles = (vmin, vmax)

            # Create render cache (everything at full resolution)
            cache = RenderCache(
                result=result,
                multiplier=settings_snapshot.multiplier,
                supersample=settings_snapshot.supersample,
                base_rgb=None,  # Will be built on-demand during postprocessing
                boundary_masks=boundary_masks,
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

                    # Upload boundary masks to GPU (avoids repeated CPU→GPU transfers on every display update)
                    if boundary_masks is not None:
                        cache.boundary_masks_gpu = []
                        for mask in boundary_masks:
                            if mask is not None:
                                cache.boundary_masks_gpu.append(GPUContext.to_gpu(mask))
                            else:
                                cache.boundary_masks_gpu.append(None)

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
                self.app.state.clear_selection()  # Clear selection when entering render mode
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
