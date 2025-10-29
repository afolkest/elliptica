"""Dear PyGui application for FlowCol."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor, Future
import math
import threading
from typing import Optional, Dict, Tuple

import numpy as np

import dearpygui.dearpygui as dpg  # type: ignore

from flowcol.app.core import AppState, RenderCache, RenderSettings
from pathlib import Path

from flowcol.app import actions
from flowcol.ui.dpg.render_modal import RenderModalController
from flowcol.ui.dpg.render_orchestrator import RenderOrchestrator
from flowcol.ui.dpg.file_io_controller import FileIOController
from flowcol.ui.dpg.cache_management_panel import CacheManagementPanel
from flowcol.ui.dpg.postprocessing_panel import PostprocessingPanel
from flowcol.ui.dpg.conductor_controls_panel import ConductorControlsPanel
from flowcol.render import array_to_pil, COLOR_PALETTES
from flowcol.types import Conductor, Project
from flowcol.pipeline import perform_render
from flowcol import defaults
from flowcol.render import colorize_array, _apply_display_transforms
# apply_region_overlays removed - using unified GPU/CPU pipeline via apply_full_postprocess_hybrid
from flowcol.postprocess.masks import rasterize_conductor_masks
from PIL import Image
from datetime import datetime



CONDUCTOR_COLORS = [
    (0.39, 0.59, 1.0, 0.7),
    (1.0, 0.39, 0.59, 0.7),
    (0.59, 1.0, 0.39, 0.7),
    (1.0, 0.78, 0.39, 0.7),
]

SUPERSAMPLE_CHOICES = defaults.SUPERSAMPLE_CHOICES
SUPERSAMPLE_LABELS = tuple(f"{value:.1f}\u00d7" for value in SUPERSAMPLE_CHOICES)
SUPERSAMPLE_LOOKUP = {label: value for label, value in zip(SUPERSAMPLE_LABELS, SUPERSAMPLE_CHOICES)}

RESOLUTION_CHOICES = defaults.RENDER_RESOLUTION_CHOICES
RESOLUTION_LABELS = tuple(f"{value:g}\u00d7" for value in RESOLUTION_CHOICES)
RESOLUTION_LOOKUP = {label: value for label, value in zip(RESOLUTION_LABELS, RESOLUTION_CHOICES)}

BACKSPACE_KEY = None
CTRL_KEY = None
C_KEY = None
V_KEY = None
if dpg is not None:
    BACKSPACE_KEY = getattr(dpg, "mvKey_Backspace", None)
    if BACKSPACE_KEY is None:
        BACKSPACE_KEY = getattr(dpg, "mvKey_Back", None)
    CTRL_KEY = getattr(dpg, "mvKey_Control", None)
    if CTRL_KEY is None:
        CTRL_KEY = getattr(dpg, "mvKey_LControl", None)
    C_KEY = getattr(dpg, "mvKey_C", None)
    V_KEY = getattr(dpg, "mvKey_V", None)


def _point_in_conductor(conductor: Conductor, x: float, y: float) -> bool:
    """Test whether canvas coordinate lands inside conductor mask."""
    cx, cy = conductor.position
    local_x = int(round(x - cx))
    local_y = int(round(y - cy))
    mask = conductor.mask
    if local_x < 0 or local_y < 0:
        return False
    if local_y >= mask.shape[0] or local_x >= mask.shape[1]:
        return False
    return mask[local_y, local_x] > 0.5


def _mask_to_rgba(mask: np.ndarray, color: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert mask to RGBA float texture."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = color[:3]
    rgba[..., 3] = mask.astype(np.float32) * color[3]
    return rgba.reshape(-1)


def _image_to_texture_data(img) -> Tuple[int, int, np.ndarray]:
    """Convert PIL image to float RGBA data."""
    img = img.convert("RGBA")
    width, height = img.size
    rgba = np.asarray(img, dtype=np.float32) / 255.0
    return width, height, rgba.reshape(-1)


def _clone_conductor(conductor: Conductor) -> Conductor:
    """Deep-copy conductor data for background rendering."""
    interior = None
    if conductor.interior_mask is not None:
        interior = conductor.interior_mask.copy()
    original = None
    if conductor.original_mask is not None:
        original = conductor.original_mask.copy()
    original_interior = None
    if conductor.original_interior_mask is not None:
        original_interior = conductor.original_interior_mask.copy()
    return Conductor(
        mask=conductor.mask.copy(),
        voltage=conductor.voltage,
        position=conductor.position,
        interior_mask=interior,
        original_mask=original,
        original_interior_mask=original_interior,
        scale_factor=conductor.scale_factor,
        blur_sigma=conductor.blur_sigma,
        blur_is_fractional=conductor.blur_is_fractional,
        smear_enabled=conductor.smear_enabled,
        smear_sigma=conductor.smear_sigma,
    )


def _snapshot_project(project: Project) -> Project:
    """Create a snapshot of project safe to use off the UI thread."""
    return Project(
        conductors=[_clone_conductor(c) for c in project.conductors],
        canvas_resolution=project.canvas_resolution,
        streamlength_factor=project.streamlength_factor,
        renders=list(project.renders),
        boundary_top=project.boundary_top,
        boundary_bottom=project.boundary_bottom,
        boundary_left=project.boundary_left,
        boundary_right=project.boundary_right,
    )


@dataclass
class FlowColApp:
    """Coordinator for Dear PyGui widgets and background work."""

    state: AppState = field(default_factory=AppState)
    state_lock: threading.RLock = field(default_factory=threading.RLock)

    canvas_id: Optional[int] = None
    canvas_layer_id: Optional[int] = None
    canvas_window_id: Optional[int] = None
    display_scale: float = 1.0
    texture_registry_id: Optional[int] = None
    colormap_registry_id: Optional[int] = None
    palette_colormaps: Dict[str, int] = field(default_factory=dict)  # palette_name -> colormap_tag
    render_texture_id: Optional[int] = None
    render_texture_size: Optional[Tuple[int, int]] = None
    viewport_created: bool = False
    edit_controls_id: Optional[int] = None
    render_controls_id: Optional[int] = None
    canvas_width_input_id: Optional[int] = None
    canvas_height_input_id: Optional[int] = None

    conductor_textures: Dict[int, int] = field(default_factory=dict)
    conductor_texture_shapes: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    canvas_dirty: bool = True

    drag_active: bool = False
    drag_last_pos: Tuple[float, float] = (0.0, 0.0)
    mouse_down_last: bool = False
    backspace_down_last: bool = False
    ctrl_c_down_last: bool = False
    ctrl_v_down_last: bool = False

    # Debouncing for expensive slider operations
    downsample_debounce_timer: Optional[threading.Timer] = None
    postprocess_debounce_timer: Optional[threading.Timer] = None
    mouse_wheel_delta: float = 0.0
    mouse_handler_registry_id: Optional[int] = None

    # Region selection for colorization
    selected_region: Optional[str] = None  # "surface" or "interior"

    # Clipboard for copy/paste
    clipboard_conductor: Optional[Conductor] = None

    def __post_init__(self) -> None:
        # Create projects directory if it doesn't exist
        projects_dir = Path.cwd() / "projects"
        projects_dir.mkdir(exist_ok=True)

        # Warmup GPU for faster first render (~750ms startup delay)
        from flowcol.gpu import GPUContext
        GPUContext.warmup()
        device_name = "MPS" if GPUContext.is_available() else "CPU"
        print(f"GPU acceleration: {device_name}")

        # Initialize controllers
        self.render_modal = RenderModalController(self)
        self.render_orchestrator = RenderOrchestrator(self)
        self.file_io = FileIOController(self)
        self.cache_panel = CacheManagementPanel(self)
        self.postprocess_panel = PostprocessingPanel(self)
        self.conductor_controls = ConductorControlsPanel(self)

        # Seed a demo conductor if project is empty so the canvas has content for manual testing.
        if not self.state.project.conductors:
            self._add_demo_conductor()

    def _on_mouse_wheel(self, sender, app_data) -> None:
        """Capture mouse wheel delta from handler."""
        self.mouse_wheel_delta = float(app_data)

    def require_backend(self) -> None:
        if dpg is None:
            raise RuntimeError("Dear PyGui is not installed. Please `pip install dearpygui` to run the GUI.")

    # ------------------------------------------------------------------
    # Building the interface
    # ------------------------------------------------------------------
    def build(self) -> None:
        """Create viewport, windows, and widgets."""
        self.require_backend()
        dpg.create_context()
        self.texture_registry_id = dpg.add_texture_registry()

        # Create colormap registry and convert our palettes to DPG colormaps
        self.colormap_registry_id = dpg.add_colormap_registry()
        for palette_name, colors_normalized in COLOR_PALETTES.items():
            # DPG expects colors as [R, G, B, A] with values 0-255
            colors_255 = [[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255] for c in colors_normalized]
            tag = f"colormap_{palette_name.replace(' ', '_').replace('&', 'and')}"
            dpg.add_colormap(colors_255, qualitative=False, tag=tag, parent=self.colormap_registry_id)
            self.palette_colormaps[palette_name] = tag

        dpg.create_viewport(title="FlowCol", width=1280, height=820)
        self.viewport_created = True

        with dpg.handler_registry() as handler_reg:
            self.mouse_handler_registry_id = handler_reg
            dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)

        # Set viewport resize callback
        dpg.set_viewport_resize_callback(self._on_viewport_resize)

        with dpg.window(label="Controls", width=360, height=-1, pos=(10, 10), tag="controls_window",
                       no_scroll_with_mouse=True):
            with dpg.group(tag="edit_controls_group") as edit_group:
                self.edit_controls_id = edit_group
                dpg.add_text("Project")
                dpg.add_button(label="Save Project...", callback=self.file_io.open_save_project_dialog, width=140)
                dpg.add_button(label="Load Project...", callback=self.file_io.open_load_project_dialog, width=140)
                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_text("Render Controls")
                dpg.add_button(label="Load Conductor...", callback=self.file_io.open_conductor_dialog)
                dpg.add_button(label="Render Field", callback=self.render_modal.open, tag="render_field_button")
                self.cache_panel.build_view_postprocessing_button(edit_group)
                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_text("Canvas Size")
                with dpg.group(horizontal=True):
                    self.canvas_width_input_id = dpg.add_input_int(
                        label="Width",
                        min_value=1,
                        min_clamped=True,
                        max_value=32768,
                        max_clamped=True,
                        step=0,
                        width=120,
                    )
                    self.canvas_height_input_id = dpg.add_input_int(
                        label="Height",
                        min_value=1,
                        min_clamped=True,
                        max_value=32768,
                        max_clamped=True,
                        step=0,
                        width=120,
                    )
                dpg.add_button(label="Apply Canvas Size", callback=self._apply_canvas_size)
                dpg.add_spacer(height=10)
                dpg.add_separator()
                self.conductor_controls.build_conductor_controls_container(edit_group)

            with dpg.group(tag="render_controls_group") as render_group:
                self.render_controls_id = render_group
                dpg.add_text("Render View")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Back to Edit", callback=self._on_back_to_edit_clicked, width=140)
                    dpg.add_button(label="Save Image", callback=self._on_save_image_clicked, width=140)
                dpg.add_spacer(height=15)
                dpg.add_separator()
                dpg.add_spacer(height=10)
                dpg.add_text("Post-processing")
                dpg.add_spacer(height=10)

                # Render Cache Status
                self.cache_panel.build_cache_status_ui(render_group)

                dpg.add_spacer(height=10)
                dpg.add_separator()
                # Build postprocessing UI (sliders, color controls, region properties)
                self.postprocess_panel.build_postprocessing_ui(render_group, self.palette_colormaps)

            dpg.add_spacer(height=10)
            dpg.add_text("Status:")
            dpg.add_text("", tag="status_text")

        # Canvas window with initial size (will be resized after viewport is shown)
        # Setting width/height prevents auto-expansion to fit drawlist
        # no_scrollbar and no_scroll_with_mouse prevent scrolling behavior
        with dpg.window(label="Canvas", pos=(380, 10), width=880, height=800, tag="canvas_window",
                       no_scrollbar=True, no_scroll_with_mouse=True) as canvas_window:
            self.canvas_window_id = canvas_window
            canvas_w, canvas_h = self.state.project.canvas_resolution
            with dpg.drawlist(width=canvas_w, height=canvas_h) as canvas:
                self.canvas_id = canvas
                # Create a draw_node for applying scale transforms
                with dpg.draw_node() as node:
                    self.canvas_layer_id = node

        self._refresh_render_texture()
        self._update_control_visibility()
        self.file_io.ensure_conductor_file_dialog()
        self._update_canvas_inputs()
        self.conductor_controls.rebuild_conductor_controls()

    # ------------------------------------------------------------------
    # Canvas window layout
    # ------------------------------------------------------------------
    def _resize_canvas_window(self) -> None:
        """Resize canvas window to fill available viewport space."""
        if dpg is None or self.canvas_window_id is None:
            return

        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()

        # Controls window is at x=10, width=360, so it takes up 370px
        # Canvas window starts at x=380 with 10px margin
        # Leave 20px margin on right side
        canvas_window_x = 380
        canvas_window_y = 10
        canvas_window_width = max(400, viewport_width - canvas_window_x - 20)
        canvas_window_height = max(300, viewport_height - canvas_window_y - 20)

        dpg.configure_item(
            self.canvas_window_id,
            width=canvas_window_width,
            height=canvas_window_height,
        )

    def _on_viewport_resize(self) -> None:
        """Handle viewport resize events."""
        self._resize_canvas_window()
        self._update_canvas_scale()

    def _update_canvas_scale(self) -> None:
        """Calculate and apply display scale transform to canvas layer."""
        if dpg is None or self.canvas_layer_id is None or self.canvas_window_id is None:
            return

        # Get window client area size (excluding titlebar, borders, scrollbars)
        window_rect = dpg.get_item_rect_size(self.canvas_window_id)
        if not window_rect:
            return
        window_w, window_h = window_rect
        if window_w <= 0 or window_h <= 0:
            return

        # Get canvas resolution
        with self.state_lock:
            canvas_w, canvas_h = self.state.project.canvas_resolution

        if canvas_w <= 0 or canvas_h <= 0:
            return

        # Calculate scale to fit canvas in window (never scale up, only down)
        scale_w = window_w / canvas_w
        scale_h = window_h / canvas_h
        scale = min(scale_w, scale_h, 1.0)

        self.display_scale = scale

        # Apply scale transform to layer
        transform = dpg.create_scale_matrix([scale, scale, 1.0])
        dpg.apply_transform(self.canvas_layer_id, transform)

    # ------------------------------------------------------------------
    # Canvas drawing
    # ------------------------------------------------------------------
    def _mark_canvas_dirty(self) -> None:
        self.canvas_dirty = True

    def _is_mouse_over_canvas(self) -> bool:
        """Check if mouse is within canvas bounds."""
        if dpg is None or self.canvas_id is None:
            return False

        # Use absolute coordinates for hit testing
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min(self.canvas_id)
        rect_max = dpg.get_item_rect_max(self.canvas_id)

        return (rect_min[0] <= mouse_x <= rect_max[0] and
                rect_min[1] <= mouse_y <= rect_max[1])

    def _get_canvas_mouse_pos(self) -> Tuple[float, float]:
        assert dpg is not None and self.canvas_id is not None
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min(self.canvas_id)
        # Get screen-space coordinates relative to canvas
        screen_x = mouse_x - rect_min[0]
        screen_y = mouse_y - rect_min[1]
        # Apply inverse scale to get canvas-space coordinates
        canvas_x = screen_x / self.display_scale if self.display_scale > 0 else screen_x
        canvas_y = screen_y / self.display_scale if self.display_scale > 0 else screen_y
        return canvas_x, canvas_y

    def _find_hit_conductor(self, x: float, y: float) -> int:
        with self.state_lock:
            conductors = self.state.project.conductors
            for idx in reversed(range(len(conductors))):
                if _point_in_conductor(conductors[idx], x, y):
                    return idx
        return -1

    def _ensure_conductor_texture(self, idx: int, mask: np.ndarray) -> int:
        assert dpg is not None and self.texture_registry_id is not None
        tex_id = self.conductor_textures.get(idx)
        width = mask.shape[1]
        height = mask.shape[0]
        existing_shape = self.conductor_texture_shapes.get(idx)
        if tex_id is not None:
            exists = dpg.does_item_exist(tex_id)
            if not exists or existing_shape != (height, width):
                if exists:
                    dpg.delete_item(tex_id)
                tex_id = None
                self.conductor_textures.pop(idx, None)

        rgba_flat = _mask_to_rgba(mask, CONDUCTOR_COLORS[idx % len(CONDUCTOR_COLORS)])

        if tex_id is None:
            tex_id = dpg.add_dynamic_texture(width, height, rgba_flat, parent=self.texture_registry_id)
            self.conductor_textures[idx] = tex_id
        else:
            dpg.set_value(tex_id, rgba_flat)
        self.conductor_texture_shapes[idx] = (height, width)
        return tex_id

    def _ensure_render_texture(self, width: int, height: int) -> int:
        assert dpg is not None and self.texture_registry_id is not None
        if self.render_texture_id is None:
            empty = np.zeros((height, width, 4), dtype=np.float32)
            empty[..., 3] = 1.0
            self.render_texture_id = dpg.add_dynamic_texture(width, height, empty.reshape(-1), parent=self.texture_registry_id)
            self.render_texture_size = (width, height)
        return self.render_texture_id

    def _add_demo_conductor(self) -> None:
        """Populate state with a simple circular conductor for quick manual testing."""
        from flowcol.app.actions import add_conductor

        canvas_w, canvas_h = self.state.project.canvas_resolution
        size = min(canvas_w, canvas_h) // 4 or 128
        y, x = np.ogrid[:size, :size]
        cy = cx = size / 2.0
        radius = size / 2.2
        mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2
        mask = mask.astype(np.float32)
        conductor = Conductor(mask=mask, voltage=1.0, position=((canvas_w - size) / 2.0, (canvas_h - size) / 2.0))
        add_conductor(self.state, conductor)

    def _apply_postprocessing(self) -> None:
        """Apply postprocessing settings to cached render and update display."""
        from scipy.ndimage import zoom

        with self.state_lock:
            cache = self.state.render_cache
            if cache is None:
                return

            settings = self.state.display_settings
            result = cache.result
            canvas_w, canvas_h = self.state.project.canvas_resolution
            target_shape = (canvas_h, canvas_w)

            # Try GPU-accelerated path
            use_gpu = False
            try:
                from flowcol.gpu import GPUContext
                from flowcol.gpu.pipeline import downsample_lic_gpu

                if GPUContext.is_available():
                    use_gpu = True
            except Exception:
                pass

            # Source for downsampling: result_gpu (GPU) or result.array (CPU)
            lic_to_process = result.array
            lic_to_process_gpu = cache.result_gpu if use_gpu else None

            # Apply downsampling with blur (GPU-accelerated)
            if use_gpu and lic_to_process_gpu is not None:
                # GPU path - much faster!
                import time
                import torch
                start = time.time()
                downsampled_gpu = downsample_lic_gpu(
                    lic_to_process_gpu,
                    target_shape,
                    settings.downsample_sigma,
                )
                torch.mps.synchronize()  # Wait for GPU work to complete
                # Uncomment for performance debugging:
                # elapsed = time.time() - start
                # print(f"🚀 GPU downsample: {elapsed*1000:.1f}ms")

                # Set GPU as primary source - CPU download happens lazily via property
                cache.set_display_array_gpu(downsampled_gpu)
            else:
                # CPU fallback
                from flowcol.render import downsample_lic
                downsampled = downsample_lic(
                    lic_to_process,
                    target_shape,
                    cache.supersample,
                    settings.downsample_sigma,
                )
                # Set CPU as primary source (clears any GPU tensor)
                cache.set_display_array_cpu(downsampled)

            # Downsample masks to match display_array resolution if needed
            if cache.conductor_masks:
                # Check if masks need downsampling by comparing mask shape to target
                first_mask = next((m for m in cache.conductor_masks if m is not None), None)
                if first_mask is not None and first_mask.shape != target_shape:
                    scale_y = target_shape[0] / first_mask.shape[0]
                    scale_x = target_shape[1] / first_mask.shape[1]
                    cache.conductor_masks = [
                        zoom(mask, (scale_y, scale_x), order=1) if mask is not None else None
                        for mask in cache.conductor_masks
                    ]
                    cache.interior_masks = [
                        zoom(mask, (scale_y, scale_x), order=1) if mask is not None else None
                        for mask in cache.interior_masks
                    ]

            # Changing display_array invalidates base_rgb
            cache.base_rgb = None

        # Update texture with new postprocessed display
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _refresh_render_texture(self) -> None:
        from PIL import Image

        if dpg is None or self.texture_registry_id is None:
            return

        with self.state_lock:
            cache = self.state.render_cache
            if cache is None or cache.display_array is None:
                # Fallback to grayscale if no render
                arr = np.zeros((32, 32), dtype=np.float32)
                pil_img = array_to_pil(arr, use_color=False)
            else:
                # Use unified GPU postprocessing pipeline (much faster!)
                from flowcol.gpu.postprocess import apply_full_postprocess_hybrid

                final_rgb = apply_full_postprocess_hybrid(
                    scalar_array=cache.display_array,
                    conductor_masks=cache.conductor_masks,
                    interior_masks=cache.interior_masks,
                    conductor_color_settings=self.state.conductor_color_settings,
                    conductors=self.state.project.conductors,
                    render_shape=cache.display_array.shape,
                    canvas_resolution=self.state.project.canvas_resolution,
                    clip_percent=self.state.display_settings.clip_percent,
                    brightness=self.state.display_settings.brightness,
                    contrast=self.state.display_settings.contrast,
                    gamma=self.state.display_settings.gamma,
                    color_enabled=self.state.display_settings.color_enabled,
                    palette=self.state.display_settings.palette,
                    lic_percentiles=cache.lic_percentiles,
                    use_gpu=True,
                    scalar_tensor=cache.display_array_gpu,  # Use pre-uploaded tensor if available
                )

                pil_img = Image.fromarray(final_rgb, mode='RGB')

        width, height, data = _image_to_texture_data(pil_img)

        if self.render_texture_id is None or self.render_texture_size != (width, height):
            if self.render_texture_id is not None:
                dpg.delete_item(self.render_texture_id)
            self.render_texture_id = dpg.add_dynamic_texture(width, height, data, parent=self.texture_registry_id)
            self.render_texture_size = (width, height)
        else:
            dpg.set_value(self.render_texture_id, data)

    def _update_canvas_inputs(self) -> None:
        if dpg is None:
            return
        with self.state_lock:
            width, height = self.state.project.canvas_resolution
        if self.canvas_width_input_id is not None:
            dpg.set_value(self.canvas_width_input_id, int(width))
        if self.canvas_height_input_id is not None:
            dpg.set_value(self.canvas_height_input_id, int(height))

    def _scale_conductor(self, idx: int, factor: float) -> bool:
        if dpg is None:
            return False
        factor = max(float(factor), 0.05)
        with self.state_lock:
            self.state.set_selected(idx)
            changed = actions.scale_conductor(self.state, idx, factor)
            if changed:
                self.conductor_textures.pop(idx, None)
                self.conductor_texture_shapes.pop(idx, None)
        if not changed:
            dpg.set_value("status_text", "Scaling limit reached.")
            return False

        self._mark_canvas_dirty()
        self.conductor_controls.update_conductor_slider_labels()
        self.postprocess_panel.update_region_properties_panel()
        dpg.set_value("status_text", f"Scaled C{idx + 1} by {factor:.2f}×")
        return True



    # ------------------------------------------------------------------
    # Canvas input processing
    # ------------------------------------------------------------------
    def _detect_region_at_point(self, canvas_x: float, canvas_y: float) -> tuple[int, Optional[str]]:
        """Detect which conductor region is at canvas point.

        Returns (conductor_idx, region) where region is "surface" or "interior" or None.
        """
        cache = self.state.render_cache
        if cache is None or cache.conductor_masks is None or cache.interior_masks is None:
            return -1, None

        # Masks are at canvas_resolution (same as canvas), so coordinates map directly
        # The render texture may be thumbnailed for display, but masks are full resolution
        mask_x = int(canvas_x)
        mask_y = int(canvas_y)

        # Check each conductor in reverse order (top to bottom)
        for idx in reversed(range(len(self.state.project.conductors))):
            if idx >= len(cache.interior_masks) or idx >= len(cache.conductor_masks):
                continue

            # Check interior first (it's inside, so higher priority)
            interior_mask = cache.interior_masks[idx]
            if interior_mask is not None:
                h, w = interior_mask.shape
                if 0 <= mask_y < h and 0 <= mask_x < w:
                    if interior_mask[mask_y, mask_x] > 0.5:
                        return idx, "interior"

            # Check surface
            surface_mask = cache.conductor_masks[idx]
            if surface_mask is not None:
                h, w = surface_mask.shape
                if 0 <= mask_y < h and 0 <= mask_x < w:
                    if surface_mask[mask_y, mask_x] > 0.5:
                        return idx, "surface"

        return -1, None

    def _process_canvas_mouse(self) -> None:
        if dpg is None or self.canvas_id is None:
            return

        mouse_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        pressed = mouse_down and not self.mouse_down_last
        released = (not mouse_down) and self.mouse_down_last

        with self.state_lock:
            mode = self.state.view_mode

        wheel_delta = self.mouse_wheel_delta
        self.mouse_wheel_delta = 0.0
        over_canvas = self._is_mouse_over_canvas()

        if mode == "edit" and over_canvas and abs(wheel_delta) > 1e-5:
            with self.state_lock:
                selected_idx = self.state.selected_idx
            # Only scale if a conductor is selected
            if selected_idx >= 0:
                scale_factor = math.exp(wheel_delta * defaults.SCROLL_SCALE_SENSITIVITY)
                scale_factor = max(0.05, min(scale_factor, 20.0))
                if self._scale_conductor(selected_idx, scale_factor):
                    x, y = self._get_canvas_mouse_pos()
                    self.drag_last_pos = (x, y)

        if mode != "edit":
            if pressed and self._is_mouse_over_canvas():
                x, y = self._get_canvas_mouse_pos()
                with self.state_lock:
                    # Use region detection in render mode for colorization
                    hit_idx, hit_region = self._detect_region_at_point(x, y)
                    self.state.set_selected(hit_idx)
                    self.selected_region = hit_region
                self.conductor_controls.update_conductor_slider_labels()
                self.postprocess_panel.update_region_properties_panel()
            self.mouse_down_last = mouse_down
            return

        if pressed and self._is_mouse_over_canvas():
            x, y = self._get_canvas_mouse_pos()
            with self.state_lock:
                project = self.state.project
                hit_idx = -1
                for idx in reversed(range(len(project.conductors))):
                    if _point_in_conductor(project.conductors[idx], x, y):
                        hit_idx = idx
                        break

                self.state.set_selected(hit_idx)
                self.selected_region = None  # Region detection only in render mode
                if hit_idx >= 0:
                    self.drag_active = True
                    self.drag_last_pos = (x, y)
                else:
                    self.drag_active = False
            self.conductor_controls.update_conductor_slider_labels()
            self.postprocess_panel.update_region_properties_panel()
            self._mark_canvas_dirty()

        if self.drag_active and mouse_down:
            x, y = self._get_canvas_mouse_pos()
            dx = x - self.drag_last_pos[0]
            dy = y - self.drag_last_pos[1]
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                with self.state_lock:
                    idx = self.state.selected_idx
                    if 0 <= idx < len(self.state.project.conductors):
                        actions.move_conductor(self.state, idx, dx, dy)
                self.drag_last_pos = (x, y)
                self._mark_canvas_dirty()

        if self.drag_active and released:
            self.drag_active = False

        self.mouse_down_last = mouse_down

    def _process_keyboard_shortcuts(self) -> None:
        if dpg is None or BACKSPACE_KEY is None:
            return

        # Only process shortcuts when mouse is over canvas (not over UI controls)
        # This prevents keyboard shortcuts from firing while typing in input fields
        over_canvas = self._is_mouse_over_canvas()
        if not over_canvas:
            self.backspace_down_last = False
            return

        # Backspace to delete
        backspace_down = dpg.is_key_down(BACKSPACE_KEY)
        if backspace_down and not self.backspace_down_last:
            with self.state_lock:
                mode = self.state.view_mode
                idx = self.state.selected_idx
                conductor_count = len(self.state.project.conductors)
                can_delete = (mode == "edit" and 0 <= idx < conductor_count)
            if can_delete:
                with self.state_lock:
                    actions.remove_conductor(self.state, idx)
                    self.conductor_textures.clear()
                    self.conductor_texture_shapes.clear()
                self._mark_canvas_dirty()
                self.conductor_controls.rebuild_conductor_controls()
                dpg.set_value("status_text", "Conductor deleted")
        self.backspace_down_last = backspace_down

        # Ctrl+C to copy
        if CTRL_KEY and C_KEY:
            ctrl_down = dpg.is_key_down(CTRL_KEY)
            c_down = dpg.is_key_down(C_KEY)
            ctrl_c_down = ctrl_down and c_down
            if ctrl_c_down and not self.ctrl_c_down_last:
                with self.state_lock:
                    mode = self.state.view_mode
                    idx = self.state.selected_idx
                    conductor_count = len(self.state.project.conductors)
                    can_copy = (mode == "edit" and 0 <= idx < conductor_count)
                if can_copy:
                    with self.state_lock:
                        self.clipboard_conductor = _clone_conductor(self.state.project.conductors[idx])
                    dpg.set_value("status_text", f"Copied C{idx + 1}")
            self.ctrl_c_down_last = ctrl_c_down

        # Ctrl+V to paste
        if CTRL_KEY and V_KEY:
            ctrl_down = dpg.is_key_down(CTRL_KEY)
            v_down = dpg.is_key_down(V_KEY)
            ctrl_v_down = ctrl_down and v_down
            if ctrl_v_down and not self.ctrl_v_down_last:
                with self.state_lock:
                    mode = self.state.view_mode
                    can_paste = (mode == "edit" and self.clipboard_conductor is not None)
                if can_paste:
                    with self.state_lock:
                        # Clone the clipboard conductor and offset it
                        pasted = _clone_conductor(self.clipboard_conductor)
                        # Offset by 30px down-right so it's visible
                        px, py = pasted.position
                        pasted.position = (px + 30.0, py + 30.0)
                        actions.add_conductor(self.state, pasted)
                        new_idx = len(self.state.project.conductors) - 1
                        self.state.set_selected(new_idx)
                    self._mark_canvas_dirty()
                    self.conductor_controls.rebuild_conductor_controls()
                    self.conductor_controls.update_conductor_slider_labels()
                    dpg.set_value("status_text", f"Pasted as C{new_idx + 1}")
            self.ctrl_v_down_last = ctrl_v_down

    def _apply_canvas_size(self, sender=None, app_data=None) -> None:
        if dpg is None:
            return

        width = int(dpg.get_value(self.canvas_width_input_id)) if self.canvas_width_input_id is not None else self.state.project.canvas_resolution[0]
        height = int(dpg.get_value(self.canvas_height_input_id)) if self.canvas_height_input_id is not None else self.state.project.canvas_resolution[1]

        width = max(1, min(width, 32768))
        height = max(1, min(height, 32768))

        with self.state_lock:
            current_size = self.state.project.canvas_resolution
            if current_size == (width, height):
                return
            actions.set_canvas_resolution(self.state, width, height)

        # Resize the actual drawlist widget to match new canvas resolution
        if self.canvas_id is not None:
            dpg.configure_item(self.canvas_id, width=width, height=height)

        self._update_canvas_inputs()
        self._mark_canvas_dirty()
        self._resize_canvas_window()  # Ensure window stays within viewport bounds
        self._update_canvas_scale()  # Recalculate display scale for new canvas size
        dpg.set_value("status_text", f"Canvas resized to {width}×{height}")

    def _on_back_to_edit_clicked(self, sender, app_data):
        with self.state_lock:
            if self.state.view_mode != "edit":
                self.state.view_mode = "edit"
                self.drag_active = False
                self._mark_canvas_dirty()
        self._update_control_visibility()
        dpg.set_value("status_text", "Edit mode.")

    def _apply_postprocessing_for_save(
        self,
        lic_array: np.ndarray,
        project: Project,
        settings,
        conductor_color_settings: dict,
        canvas_resolution: tuple[int, int],
        margin_physical: float,
        scale: float,
        offset_x: int,
        offset_y: int,
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
        from flowcol.postprocess.masks import rasterize_conductor_masks

        # Generate masks at this resolution
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
            vmin = float(np.percentile(lic_array, 0.5))
            vmax = float(np.percentile(lic_array, 99.5))
            lic_percentiles = (vmin, vmax)

        # Use unified GPU postprocessing pipeline (or CPU fallback)
        from flowcol.gpu.postprocess import apply_full_postprocess_hybrid

        final_rgb = apply_full_postprocess_hybrid(
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

    def _on_save_image_clicked(self, sender, app_data):
        """Save the final rendered image to disk.

        If supersampled: saves two versions (_supersampled and _final).
        If not supersampled: saves one version.
        """
        if dpg is None:
            return

        with self.state_lock:
            cache = self.state.render_cache
            if cache is None or cache.result is None:
                dpg.set_value("status_text", "No render to save.")
                return

            # Snapshot everything
            result = cache.result
            project = _snapshot_project(self.state.project)
            settings = replace(self.state.display_settings)
            conductor_color_settings = {k: v for k, v in self.state.conductor_color_settings.items()}
            multiplier = cache.multiplier
            supersample = cache.supersample

        from flowcol.render import downsample_lic
        from PIL import Image
        from datetime import datetime

        canvas_w, canvas_h = project.canvas_resolution
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path.cwd() / "outputs"
        output_dir.mkdir(exist_ok=True)
        margin_physical = result.margin

        # Use render result directly (edge blur removed)
        lic_array = result.array

        if supersample > 1.0:
            # Save supersampled version at render resolution
            render_scale = multiplier * supersample
            final_rgb_super = self._apply_postprocessing_for_save(
                lic_array,
                project,
                settings,
                conductor_color_settings,
                (canvas_w, canvas_h),
                margin_physical,
                render_scale,
                result.offset_x,
                result.offset_y,
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

            final_rgb_output = self._apply_postprocessing_for_save(
                downsampled_lic,
                project,
                settings,
                conductor_color_settings,
                (canvas_w, canvas_h),
                margin_physical,
                output_scale,
                output_offset_x,
                output_offset_y,
            )

            h_output, w_output = final_rgb_output.shape[:2]
            output_path_final = output_dir / f"flowcol_{w_output}x{h_output}_final_{timestamp}.png"
            pil_img_output = Image.fromarray(final_rgb_output, mode='RGB')
            pil_img_output.save(output_path_final)

            dpg.set_value("status_text", f"Saved {output_path_super.name} and {output_path_final.name}")
        else:
            # No supersampling: save single version
            render_scale = multiplier
            final_rgb = self._apply_postprocessing_for_save(
                lic_array,
                project,
                settings,
                conductor_color_settings,
                (canvas_w, canvas_h),
                margin_physical,
                render_scale,
                result.offset_x,
                result.offset_y,
            )

            h, w = final_rgb.shape[:2]
            output_path = output_dir / f"flowcol_{w}x{h}_{timestamp}.png"
            pil_img = Image.fromarray(final_rgb, mode='RGB')
            pil_img.save(output_path)

            dpg.set_value("status_text", f"Saved {output_path.name}")

    def _redraw_canvas(self) -> None:
        if dpg is None or self.canvas_layer_id is None:
            return
        self.canvas_dirty = False

        self._refresh_render_texture()

        with self.state_lock:
            project = self.state.project
            selected_idx = self.state.selected_idx
            canvas_w, canvas_h = project.canvas_resolution
            conductors = list(project.conductors)
            render_cache = self.state.render_cache
            view_mode = self.state.view_mode

        # Clear the layer (transform persists on layer)
        dpg.delete_item(self.canvas_layer_id, children_only=True)

        # Draw background
        dpg.draw_rectangle((0, 0), (canvas_w, canvas_h), color=(60, 60, 60, 255), fill=(20, 20, 20, 255), parent=self.canvas_layer_id)

        # Draw grid in edit mode (based on physical units, not pixels)
        if view_mode == "edit":
            # Grid every 0.1 canvas units (10% of canvas dimension)
            grid_fraction = 0.1
            grid_spacing_x = canvas_w * grid_fraction
            grid_spacing_y = canvas_h * grid_fraction
            mid_x = canvas_w / 2.0
            mid_y = canvas_h / 2.0

            # Vertical grid lines
            x = 0.0
            while x <= canvas_w:
                is_midline = abs(x - mid_x) < 0.5
                color = (120, 120, 140, 180) if is_midline else (50, 50, 50, 100)
                thickness = 2.5 if is_midline else 1.0
                dpg.draw_line((x, 0), (x, canvas_h), color=color, thickness=thickness, parent=self.canvas_layer_id)
                x += grid_spacing_x

            # Horizontal grid lines
            y = 0.0
            while y <= canvas_h:
                is_midline = abs(y - mid_y) < 0.5
                color = (120, 120, 140, 180) if is_midline else (50, 50, 50, 100)
                thickness = 2.5 if is_midline else 1.0
                dpg.draw_line((0, y), (canvas_w, y), color=color, thickness=thickness, parent=self.canvas_layer_id)
                y += grid_spacing_y

        if view_mode == "render" and render_cache and self.render_texture_id is not None and self.render_texture_size:
            tex_w, tex_h = self.render_texture_size
            if tex_w > 0 and tex_h > 0:
                scale_x = canvas_w / tex_w
                scale_y = canvas_h / tex_h
                pmax = (tex_w * scale_x, tex_h * scale_y)
                dpg.draw_image(
                    self.render_texture_id,
                    (0, 0),
                    pmax,
                    uv_min=(0.0, 0.0),
                    uv_max=(1.0, 1.0),
                    parent=self.canvas_layer_id,
                )

        if view_mode == "edit" or render_cache is None:
            for idx, conductor in enumerate(conductors):
                tex_id = self._ensure_conductor_texture(idx, conductor.mask)
                x0, y0 = conductor.position
                width = conductor.mask.shape[1]
                height = conductor.mask.shape[0]
                dpg.draw_image(
                    tex_id,
                    pmin=(x0, y0),
                    pmax=(x0 + width, y0 + height),
                    uv_min=(0.0, 0.0),
                    uv_max=(1.0, 1.0),
                    parent=self.canvas_layer_id,
                )
                if idx == selected_idx:
                    dpg.draw_rectangle(
                        (x0, y0),
                        (x0 + width, y0 + height),
                        color=(255, 255, 100, 200),
                        thickness=2.0,
                        parent=self.canvas_layer_id,
                    )

    # ------------------------------------------------------------------
    # Canvas input processing
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def render(self) -> None:
        """Run Dear PyGui event loop."""
        self.require_backend()
        if not self.viewport_created:
            self.build()

        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Resize canvas window and update scale after viewport is shown and has valid dimensions
        self._resize_canvas_window()
        self._update_canvas_scale()

        try:
            while dpg.is_dearpygui_running():
                self._process_canvas_mouse()
                self._process_keyboard_shortcuts()
                self.render_orchestrator.poll()
                if self.canvas_dirty:
                    self._redraw_canvas()
                dpg.render_dearpygui_frame()
        finally:
            self.render_orchestrator.shutdown(wait=False)
            dpg.destroy_context()


    def _update_control_visibility(self) -> None:
        if dpg is None:
            return
        with self.state_lock:
            mode = self.state.view_mode
            has_cache = self.state.render_cache is not None

        if self.edit_controls_id is not None:
            dpg.configure_item(self.edit_controls_id, show=(mode == "edit"))
        if self.render_controls_id is not None:
            dpg.configure_item(self.render_controls_id, show=(mode == "render"))

        # In edit mode: always show "Render Field", add "View Postprocessing" if cache exists
        if dpg.does_item_exist("render_field_button"):
            dpg.configure_item("render_field_button", show=(mode == "edit"))

        if self.cache_panel.view_postprocessing_button_id is not None:
            show_button = (mode == "edit" and has_cache)
            dpg.configure_item(self.cache_panel.view_postprocessing_button_id, show=show_button)

        self.conductor_controls.update_conductor_slider_labels()

def run() -> None:
    """Launch FlowCol Dear PyGui application."""
    FlowColApp().render()
