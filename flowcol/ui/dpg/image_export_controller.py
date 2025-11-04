"""Image export controller for FlowCol UI.

Handles saving rendered images to disk with support for:
- Supersampled exports (saves 2 versions)
- Arbitrary resolution exports
- Postprocessing pipeline at export resolution
"""

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from datetime import datetime

from flowcol.postprocess.masks import rasterize_conductor_masks
from flowcol.render import downsample_lic

if TYPE_CHECKING:
    from flowcol.ui.dpg.app import FlowColApp, _snapshot_project
    from flowcol.types import Project

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class ImageExportController:
    """Controller for image export operations."""

    def __init__(self, app: "FlowColApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main FlowColApp instance
        """
        self.app = app

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
            from flowcol.ui.dpg.app import _snapshot_project
            result = cache.result
            project = _snapshot_project(self.app.state.project)
            settings = replace(self.app.state.display_settings)
            conductor_color_settings = {k: v for k, v in self.app.state.conductor_color_settings.items()}
            multiplier = cache.multiplier
            supersample = cache.supersample
            # Snapshot cached masks (these are correct - don't regenerate!)
            cached_conductor_masks = [m.copy() if m is not None else None for m in cache.conductor_masks] if cache.conductor_masks else None
            cached_interior_masks = [m.copy() if m is not None else None for m in cache.interior_masks] if cache.interior_masks else None

            # DEBUG: Print conductor color settings
            print("\n=== EXPORT DEBUG ===")
            print(f"Total conductors in project: {len(project.conductors)}")
            print(f"conductor_color_settings keys: {list(conductor_color_settings.keys())}")
            for conductor in project.conductors:
                print(f"Conductor ID: {conductor.id} (type: {type(conductor.id)})")
                if conductor.id in conductor_color_settings:
                    cs = conductor_color_settings[conductor.id]
                    print(f"  Interior enabled: {cs.interior.enabled}")
                    print(f"  Interior palette: {cs.interior.palette}")
                    print(f"  Interior use_palette: {cs.interior.use_palette}")
                    print(f"  Surface enabled: {cs.surface.enabled}")
                else:
                    print(f"  ⚠️  NOT FOUND in conductor_color_settings!")
            print("===================\n")

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
                conductor_color_settings,
                (canvas_w, canvas_h),
                margin_physical,
                render_scale,
                result.offset_x,
                result.offset_y,
                cached_conductor_masks,
                cached_interior_masks,
            )

            h_super, w_super = final_rgb_super.shape[:2]
            output_path_super = output_dir / f"flowcol_{w_super}x{h_super}_supersampled_{timestamp}.png"
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
            final_rgb_output = self._apply_postprocessing_at_resolution(
                downsampled_lic,
                project,
                settings,
                conductor_color_settings,
                (canvas_w, canvas_h),
                margin_physical,
                output_scale,
                output_offset_x,
                output_offset_y,
                None,  # Regenerate masks for downsampled resolution
                None,
            )

            h_output, w_output = final_rgb_output.shape[:2]
            output_path_final = output_dir / f"flowcol_{w_output}x{h_output}_final_{timestamp}.png"
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
                conductor_color_settings,
                (canvas_w, canvas_h),
                margin_physical,
                render_scale,
                result.offset_x,
                result.offset_y,
                cached_conductor_masks,
                cached_interior_masks,
            )

            h, w = final_rgb.shape[:2]
            output_path = output_dir / f"flowcol_{w}x{h}_{timestamp}.png"
            pil_img = Image.fromarray(final_rgb, mode='RGB')
            pil_img.save(output_path)

            dpg.set_value("status_text", f"Saved {output_path.name}")

    def _apply_postprocessing_at_resolution(
        self,
        lic_array: np.ndarray,
        project: "Project",
        settings,
        conductor_color_settings: dict,
        canvas_resolution: tuple[int, int],
        margin_physical: float,
        scale: float,
        offset_x: int,
        offset_y: int,
        cached_conductor_masks: list = None,
        cached_interior_masks: list = None,
    ) -> np.ndarray:
        """Apply full post-processing pipeline to LIC array at any resolution.

        Args:
            lic_array: Grayscale LIC array
            project: Project snapshot
            settings: Display settings snapshot
            conductor_color_settings: Conductor color settings
            canvas_resolution: Canvas resolution (width, height)
            margin_physical: Physical margin in canvas units
            scale: Pixels per canvas unit (multiplier or multiplier*supersample)
            offset_x: Crop offset X from render result
            offset_y: Crop offset Y from render result

        Returns:
            Final RGB array with all post-processing applied
        """
        # Use cached masks if available (avoids regeneration errors), otherwise generate at this resolution
        if cached_conductor_masks is not None and cached_interior_masks is not None:
            conductor_masks = cached_conductor_masks
            interior_masks = cached_interior_masks
        else:
            conductor_masks = None
            interior_masks = None
            if project.conductors:
                conductor_masks, interior_masks = rasterize_conductor_masks(
                    project.conductors,
                    lic_array.shape,
                    margin_physical,
                    scale,
                    offset_x,
                    offset_y,
                )

        # Compute percentiles for smear (if needed at export resolution)
        lic_percentiles = None
        if any(c.smear_enabled for c in project.conductors):
            clip_percent = float(settings.clip_percent)
            if clip_percent > 0.0:
                vmin = float(np.percentile(lic_array, clip_percent))
                vmax = float(np.percentile(lic_array, 100.0 - clip_percent))
            else:
                vmin = float(np.min(lic_array))
                vmax = float(np.max(lic_array))
            lic_percentiles = (vmin, vmax)

        # Use unified GPU postprocessing pipeline (or CPU fallback)
        from flowcol.gpu.postprocess import apply_full_postprocess_hybrid

        final_rgb, _ = apply_full_postprocess_hybrid(
            scalar_array=lic_array,
            conductor_masks=conductor_masks,
            interior_masks=interior_masks,
            conductor_color_settings=conductor_color_settings,
            conductors=project.conductors,
            render_shape=lic_array.shape,
            canvas_resolution=canvas_resolution,
            clip_percent=settings.clip_percent,
            brightness=settings.brightness,
            contrast=settings.contrast,
            gamma=settings.gamma,
            color_enabled=settings.color_enabled,
            palette=settings.palette,
            lic_percentiles=lic_percentiles,
            use_gpu=True,  # GPU acceleration for faster exports
            scalar_tensor=None,  # Will upload on-demand
        )

        return final_rgb
