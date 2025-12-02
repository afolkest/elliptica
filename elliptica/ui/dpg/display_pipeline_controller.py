"""Display pipeline controller for Elliptica UI.

Manages the real-time display pipeline:
- Display cache invalidation (base_rgb)
- Texture refreshes
- Coordinating postprocessing updates
- Async postprocessing to keep UI responsive
"""

from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class DisplayPipelineController:
    """Controller for display pipeline and postprocessing coordination.

    Owns the TextureManager and provides clean API for display updates.
    Runs heavy postprocessing in a background thread to keep UI responsive.
    """

    def __init__(self, app: "EllipticaApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main EllipticaApp instance
        """
        self.app = app

        # Import here to avoid circular dependency
        from elliptica.ui.dpg.texture_manager import TextureManager
        self.texture_manager = TextureManager(app)

        # Async postprocessing state
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self.postprocess_future: Optional[Future] = None
        self._pending_invalidate: bool = False

    def refresh_display(self, invalidate_cache: bool = False) -> None:
        """Refresh the display with current postprocessing settings.

        Runs postprocessing asynchronously to keep UI responsive.

        Args:
            invalidate_cache: If True, invalidate base_rgb cache before refresh
        """
        # If a postprocess job is already running, mark that we need another refresh
        # (settings may have changed while computing)
        if self.postprocess_future is not None and not self.postprocess_future.done():
            self._pending_invalidate = self._pending_invalidate or invalidate_cache
            return

        if invalidate_cache:
            with self.app.state_lock:
                cache = self.app.state.render_cache
                if cache is not None:
                    cache.base_rgb = None

        # Start async postprocessing
        self._start_postprocess_job()

    def _start_postprocess_job(self) -> None:
        """Submit postprocessing work to background thread."""
        # Snapshot all settings needed for postprocessing under the lock
        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None or cache.result is None:
                # No render to postprocess - do sync fallback
                self.texture_manager.refresh_render_texture()
                self.app.canvas_renderer.mark_dirty()
                return

            # Snapshot settings (these are simple values, safe to copy)
            settings_snapshot = {
                'clip_percent': self.app.state.display_settings.clip_percent,
                'brightness': self.app.state.display_settings.brightness,
                'contrast': self.app.state.display_settings.contrast,
                'gamma': self.app.state.display_settings.gamma,
                'color_enabled': self.app.state.display_settings.color_enabled,
                'palette': self.app.state.display_settings.palette,
                'lightness_expr': self.app.state.display_settings.lightness_expr,
                'saturation': self.app.state.display_settings.saturation,
                'canvas_resolution': self.app.state.project.canvas_resolution,
            }

            # Deep copy conductor_color_settings (mutable nested dict)
            from copy import deepcopy
            conductor_color_settings_snapshot = deepcopy(self.app.state.conductor_color_settings)
            conductors_snapshot = list(self.app.state.project.conductors)
            color_config_snapshot = self.app.state.color_config

            # Get cached percentiles if valid
            lic_percentiles = None
            if cache.lic_percentiles is not None:
                cached_clip = cache.lic_percentiles_clip_percent
                if cached_clip is not None and abs(cached_clip - settings_snapshot['clip_percent']) < 0.01:
                    lic_percentiles = cache.lic_percentiles

            # References to GPU tensors (safe - they're not modified during postprocess)
            scalar_array = cache.result.array
            render_shape = cache.result.array.shape
            conductor_masks = cache.conductor_masks
            interior_masks = cache.interior_masks
            scalar_tensor = cache.result_gpu
            conductor_masks_gpu = cache.conductor_masks_gpu
            interior_masks_gpu = cache.interior_masks_gpu
            ex_tensor = cache.ex_gpu
            ey_tensor = cache.ey_gpu

            # Get or compute resized solution tensors (cached to avoid non-determinism)
            solution_gpu = None
            if cache.solution_gpu and scalar_tensor is not None:
                target_shape = scalar_tensor.shape
                # Check if we have cached resized tensors at the right shape
                if (cache.solution_gpu_resized is not None and
                    cache.solution_gpu_lic_shape == target_shape):
                    solution_gpu = cache.solution_gpu_resized
                else:
                    # Resize and cache
                    import torch
                    resized = {}
                    for name, tensor in cache.solution_gpu.items():
                        if tensor.shape == target_shape:
                            resized[name] = tensor
                        else:
                            tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
                            r = torch.nn.functional.interpolate(
                                tensor_4d, size=target_shape, mode='bilinear', align_corners=False
                            )
                            resized[name] = r.squeeze(0).squeeze(0)
                    cache.solution_gpu_resized = resized
                    cache.solution_gpu_lic_shape = target_shape
                    solution_gpu = resized

        def job():
            """Background postprocessing job."""
            from elliptica.gpu.postprocess import apply_full_postprocess_hybrid
            from elliptica.expr import ExprError

            try:
                final_rgb, used_percentiles = apply_full_postprocess_hybrid(
                    scalar_array=scalar_array,
                    conductor_masks=conductor_masks,
                    interior_masks=interior_masks,
                    conductor_color_settings=conductor_color_settings_snapshot,
                    conductors=conductors_snapshot,
                    render_shape=render_shape,
                    canvas_resolution=settings_snapshot['canvas_resolution'],
                    clip_percent=settings_snapshot['clip_percent'],
                    brightness=settings_snapshot['brightness'],
                    contrast=settings_snapshot['contrast'],
                    gamma=settings_snapshot['gamma'],
                    color_enabled=settings_snapshot['color_enabled'],
                    palette=settings_snapshot['palette'],
                    lic_percentiles=lic_percentiles,
                    use_gpu=True,
                    scalar_tensor=scalar_tensor,
                    conductor_masks_gpu=conductor_masks_gpu,
                    interior_masks_gpu=interior_masks_gpu,
                    color_config=color_config_snapshot,
                    ex_tensor=ex_tensor,
                    ey_tensor=ey_tensor,
                    lightness_expr=settings_snapshot['lightness_expr'],
                    solution_gpu=solution_gpu,
                    saturation=settings_snapshot['saturation'],
                )
                return ('success', final_rgb, used_percentiles, settings_snapshot['clip_percent'])
            except ExprError as e:
                return ('expr_error', str(e))
            except Exception as e:
                return ('error', str(e))

        self.postprocess_future = self.executor.submit(job)

    def poll(self) -> None:
        """Poll for postprocess job completion. Call from main loop."""
        if self.postprocess_future is None:
            return
        if not self.postprocess_future.done():
            return

        # Job completed - get result and update texture on main thread
        try:
            result = self.postprocess_future.result()
        except Exception as e:
            result = ('error', str(e))

        self.postprocess_future = None

        if result[0] == 'success':
            _, final_rgb, used_percentiles, clip_percent = result

            # Update cache with computed percentiles
            with self.app.state_lock:
                cache = self.app.state.render_cache
                if cache is not None:
                    cache.lic_percentiles = used_percentiles
                    cache.lic_percentiles_clip_percent = clip_percent

            # Update texture (must be on main thread)
            self.texture_manager.update_texture_from_rgb(final_rgb)
            self.app.canvas_renderer.mark_dirty()

            # Clear any expression error
            if dpg is not None and dpg.does_item_exist("expr_error_text"):
                pass  # Let panel manage error display

        elif result[0] == 'expr_error':
            # Show expression error but keep previous display
            print(f"⚠️  Expression error: {result[1]}")
            if dpg is not None and dpg.does_item_exist("expr_error_text"):
                dpg.set_value("expr_error_text", f"Error: {result[1]}")

        elif result[0] == 'error':
            # Log general postprocess errors
            print(f"⚠️  Postprocess error: {result[1]}")

        # If another refresh was requested while we were computing, start a new job
        if self._pending_invalidate:
            self._pending_invalidate = False
            self.refresh_display(invalidate_cache=True)

    def invalidate_and_refresh(self) -> None:
        """Invalidate display cache and refresh (for settings that affect base colorization)."""
        self.refresh_display(invalidate_cache=True)

    def shutdown(self) -> None:
        """Shutdown the executor thread pool."""
        self.executor.shutdown(wait=False)
