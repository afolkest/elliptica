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
from flowcol.render import array_to_pil, COLOR_PALETTES
from flowcol.types import Conductor, Project
from flowcol.pipeline import perform_render
from flowcol import defaults
from flowcol.mask_utils import load_conductor_masks
from flowcol.render import colorize_array, apply_conductor_smear, _apply_display_transforms
from flowcol.postprocess.color import apply_region_overlays
from flowcol.postprocess.masks import rasterize_conductor_masks
from flowcol.serialization import save_project, load_project, save_render_cache, load_render_cache, compute_project_fingerprint
from PIL import Image
from datetime import datetime



CONDUCTOR_COLORS = [
    (0.39, 0.59, 1.0, 0.7),
    (1.0, 0.39, 0.59, 0.7),
    (0.59, 1.0, 0.39, 0.7),
    (1.0, 0.78, 0.39, 0.7),
]

MAX_CANVAS_DIM = 8192

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


def _label_for_supersample(value: float) -> str:
    idx = min(range(len(SUPERSAMPLE_CHOICES)), key=lambda i: abs(SUPERSAMPLE_CHOICES[i] - value))
    return SUPERSAMPLE_LABELS[idx]


def _label_for_multiplier(value: float) -> str:
    idx = min(range(len(RESOLUTION_CHOICES)), key=lambda i: abs(RESOLUTION_CHOICES[i] - value))
    return RESOLUTION_LABELS[idx]


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
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1))

    render_future: Optional[Future] = None
    render_error: Optional[str] = None

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
    render_modal_id: Optional[int] = None
    render_supersample_radio_id: Optional[int] = None
    render_multiplier_radio_id: Optional[int] = None
    render_passes_input_id: Optional[int] = None
    render_streamlength_input_id: Optional[int] = None
    render_margin_input_id: Optional[int] = None
    render_seed_input_id: Optional[int] = None
    render_sigma_input_id: Optional[int] = None
    boundary_top_checkbox_id: Optional[int] = None
    boundary_bottom_checkbox_id: Optional[int] = None
    boundary_left_checkbox_id: Optional[int] = None
    boundary_right_checkbox_id: Optional[int] = None
    canvas_width_input_id: Optional[int] = None
    canvas_height_input_id: Optional[int] = None
    conductor_file_dialog_id: Optional[str] = None
    save_project_dialog_id: Optional[str] = None
    load_project_dialog_id: Optional[str] = None
    conductor_controls_container_id: Optional[int] = None
    conductor_slider_ids: Dict[int, int] = field(default_factory=dict)
    postprocess_downsample_slider_id: Optional[int] = None
    postprocess_clip_slider_id: Optional[int] = None
    postprocess_brightness_slider_id: Optional[int] = None
    postprocess_contrast_slider_id: Optional[int] = None
    postprocess_gamma_slider_id: Optional[int] = None
    color_enabled_checkbox_id: Optional[int] = None

    # Cache status display
    cache_status_text_id: Optional[int] = None
    cache_warning_group_id: Optional[int] = None
    back_to_edit_button_id: Optional[int] = None
    mark_clean_button_id: Optional[int] = None
    discard_cache_button_id: Optional[int] = None
    view_postprocessing_button_id: Optional[int] = None

    # Current project file path (for auto-saving cache)
    current_project_path: Optional[str] = None

    conductor_textures: Dict[int, int] = field(default_factory=dict)
    conductor_texture_shapes: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    canvas_dirty: bool = True

    drag_active: bool = False
    drag_last_pos: Tuple[float, float] = (0.0, 0.0)
    mouse_down_last: bool = False
    render_modal_open: bool = False
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
                dpg.add_button(label="Save Project...", callback=self._open_save_project_dialog, width=140)
                dpg.add_button(label="Load Project...", callback=self._open_load_project_dialog, width=140)
                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_text("Render Controls")
                dpg.add_button(label="Load Conductor...", callback=self._open_conductor_dialog)
                dpg.add_button(label="Render Field", callback=self._open_render_modal, tag="render_field_button")
                self.view_postprocessing_button_id = dpg.add_button(
                    label="View Postprocessing",
                    callback=self._on_view_postprocessing_clicked,
                    show=False
                )
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
                dpg.add_text("Conductor Voltages")
                self.conductor_controls_container_id = dpg.add_child_window(
                    autosize_x=True,
                    height=400,
                    border=False,
                    tag="conductor_controls_child",
                    no_scroll_with_mouse=False,
                )

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
                dpg.add_text("Render Cache")
                self.cache_status_text_id = dpg.add_text("No cached render")
                with dpg.group() as cache_warning_group:
                    self.cache_warning_group_id = cache_warning_group
                    dpg.add_text("⚠️  Project modified since render", color=(255, 200, 100))
                    with dpg.group(horizontal=True):
                        self.mark_clean_button_id = dpg.add_button(
                            label="Mark Clean",
                            callback=self._on_mark_clean_clicked,
                            width=90
                        )
                        self.discard_cache_button_id = dpg.add_button(
                            label="Discard",
                            callback=self._on_discard_cache_clicked,
                            width=90
                        )
                dpg.configure_item(self.cache_warning_group_id, show=False)

                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_spacer(height=10)

                self.postprocess_downsample_slider_id = dpg.add_slider_float(
                    label="Downsampling Blur",
                    default_value=self.state.display_settings.downsample_sigma,
                    min_value=0.0,
                    max_value=2.0,
                    format="%.2f",
                    callback=self._on_downsample_slider,
                    width=200,
                )

                self.postprocess_clip_slider_id = dpg.add_slider_float(
                    label="Clip %",
                    default_value=self.state.display_settings.clip_percent,
                    min_value=0.0,
                    max_value=defaults.MAX_CLIP_PERCENT,
                    format="%.2f%%",
                    callback=self._on_clip_slider,
                    width=200,
                )

                self.postprocess_brightness_slider_id = dpg.add_slider_float(
                    label="Brightness",
                    default_value=self.state.display_settings.brightness,
                    min_value=-0.5,
                    max_value=0.5,
                    format="%.2f",
                    callback=self._on_brightness_slider,
                    width=200,
                )

                self.postprocess_contrast_slider_id = dpg.add_slider_float(
                    label="Contrast",
                    default_value=self.state.display_settings.contrast,
                    min_value=0.5,
                    max_value=2.0,
                    format="%.2f",
                    callback=self._on_contrast_slider,
                    width=200,
                )

                self.postprocess_gamma_slider_id = dpg.add_slider_float(
                    label="Gamma",
                    default_value=self.state.display_settings.gamma,
                    min_value=0.3,
                    max_value=3.0,
                    format="%.2f",
                    callback=self._on_gamma_slider,
                    width=200,
                )

                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_text("Colorization")
                dpg.add_spacer(height=10)

                self.color_enabled_checkbox_id = dpg.add_checkbox(
                    label="Enable Color",
                    default_value=self.state.display_settings.color_enabled,
                    callback=self._on_color_enabled,
                )

                from flowcol.render import list_color_palettes
                palette_names = list(list_color_palettes())

                # Global palette selection with popup menu
                dpg.add_text("Global Palette")
                global_palette_button = dpg.add_button(
                    label="Choose Palette...",
                    width=200,
                    tag="global_palette_button",
                )
                dpg.add_text(
                    f"Current: {self.state.display_settings.palette}",
                    tag="global_palette_current_text"
                )

                # Popup menu for global palette selection
                with dpg.popup(global_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="global_palette_popup"):
                    dpg.add_text("Select a palette:")
                    dpg.add_separator()
                    with dpg.child_window(width=380, height=300):
                        for palette_name in palette_names:
                            colormap_tag = self.palette_colormaps[palette_name]
                            btn = dpg.add_colormap_button(
                                label=palette_name,
                                width=350,
                                height=25,
                                callback=self._on_global_palette_button,
                                user_data=palette_name,
                                tag=f"global_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                            )
                            dpg.bind_colormap(btn, colormap_tag)

                dpg.add_spacer(height=10)
                dpg.add_separator()

                # Region properties (shown when conductor selected in render mode)
                with dpg.collapsing_header(label="Region Properties", default_open=True, tag="region_properties_header") as region_header:
                    dpg.add_text("Select a conductor region to customize", tag="region_hint_text")

                    dpg.add_spacer(height=5)
                    dpg.add_text("Surface (Field Lines)", tag="surface_label")
                    self.surface_enabled_checkbox_id = dpg.add_checkbox(
                        label="Enable Custom Palette",
                        callback=self._on_surface_enabled,
                        tag="surface_enabled_checkbox",
                    )
                    # Surface palette popup
                    surface_palette_button = dpg.add_button(
                        label="Choose Surface Palette...",
                        width=200,
                        tag="surface_palette_button",
                    )
                    dpg.add_text("Current: None", tag="surface_palette_current_text")

                    with dpg.popup(surface_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="surface_palette_popup"):
                        dpg.add_text("Select surface palette:")
                        dpg.add_separator()
                        with dpg.child_window(width=380, height=250):
                            for palette_name in palette_names:
                                colormap_tag = self.palette_colormaps[palette_name]
                                btn = dpg.add_colormap_button(
                                    label=palette_name,
                                    width=350,
                                    height=25,
                                    callback=self._on_surface_palette_button,
                                    user_data=palette_name,
                                    tag=f"surface_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                                )
                                dpg.bind_colormap(btn, colormap_tag)

                    dpg.add_spacer(height=10)
                    dpg.add_text("Interior (Hollow Region)", tag="interior_label")
                    self.interior_enabled_checkbox_id = dpg.add_checkbox(
                        label="Enable Custom Palette",
                        callback=self._on_interior_enabled,
                        tag="interior_enabled_checkbox",
                    )
                    # Interior palette popup
                    interior_palette_button = dpg.add_button(
                        label="Choose Interior Palette...",
                        width=200,
                        tag="interior_palette_button",
                    )
                    dpg.add_text("Current: None", tag="interior_palette_current_text")

                    with dpg.popup(interior_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="interior_palette_popup"):
                        dpg.add_text("Select interior palette:")
                        dpg.add_separator()
                        with dpg.child_window(width=380, height=250):
                            for palette_name in palette_names:
                                colormap_tag = self.palette_colormaps[palette_name]
                                btn = dpg.add_colormap_button(
                                    label=palette_name,
                                    width=350,
                                    height=25,
                                    callback=self._on_interior_palette_button,
                                    user_data=palette_name,
                                    tag=f"interior_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                                )
                                dpg.bind_colormap(btn, colormap_tag)

                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_text("Interior Smear")
                    self.smear_enabled_checkbox_id = dpg.add_checkbox(
                        label="Enable Interior Smear",
                        callback=self._on_smear_enabled,
                        tag="smear_enabled_checkbox",
                    )
                    self.smear_sigma_slider_id = dpg.add_slider_float(
                        label="Blur Sigma",
                        min_value=0.1,
                        max_value=10.0,
                        format="%.1f px",
                        callback=self._on_smear_sigma,
                        tag="smear_sigma_slider",
                        width=200,
                    )

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
        self._ensure_conductor_file_dialog()
        self._update_canvas_inputs()
        self._rebuild_conductor_controls()

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
        from flowcol.app.actions import ensure_base_rgb
        from flowcol.postprocess.color import apply_region_overlays
        from PIL import Image

        if dpg is None or self.texture_registry_id is None:
            return

        with self.state_lock:
            # Build base_rgb if needed
            if not ensure_base_rgb(self.state):
                # Fallback to grayscale if no render
                arr = np.zeros((32, 32), dtype=np.float32)
                pil_img = array_to_pil(arr, use_color=False)
            else:
                # Use cached base_rgb
                cache = self.state.render_cache
                base_rgb = cache.base_rgb.copy()

                # Apply conductor smear effect (post-processing on RGB)
                from flowcol.render import apply_conductor_smear
                if any(c.smear_enabled for c in self.state.project.conductors):
                    base_rgb = apply_conductor_smear(
                        base_rgb,
                        cache.display_array,  # LIC grayscale for re-blurring
                        self.state.project,
                        self.state.display_settings.palette,
                        cache.display_array.shape,  # (height, width)
                        color_enabled=self.state.display_settings.color_enabled,
                        lic_percentiles=cache.lic_percentiles,  # Precomputed for performance
                    )

                # Apply per-region overlays
                if cache.conductor_masks and cache.interior_masks:
                    final_rgb = apply_region_overlays(
                        base_rgb,
                        cache.display_array,
                        cache.conductor_masks,
                        cache.interior_masks,
                        self.state.conductor_color_settings,
                        self.state.project.conductors,
                        self.state.display_settings.to_color_params(),
                        cache.display_array_gpu,  # GPU acceleration for palette colorization
                    )
                else:
                    final_rgb = base_rgb

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

    def _rebuild_conductor_controls(self) -> None:
        if dpg is None or self.conductor_controls_container_id is None:
            return

        dpg.delete_item(self.conductor_controls_container_id, children_only=True)
        self.conductor_slider_ids.clear()

        with self.state_lock:
            conductors = list(self.state.project.conductors)
            selected_idx = self.state.selected_idx

        if not conductors:
            dpg.add_text("No conductors loaded.", parent=self.conductor_controls_container_id)
            return

        for idx, conductor in enumerate(conductors):
            label = f"C{idx + 1}"
            if idx == selected_idx:
                label += " (selected)"
            slider_id = dpg.add_slider_float(
                label=label,
                default_value=float(conductor.voltage),
                min_value=-1.0,
                max_value=1.0,
                format="%.3f",
                callback=self._on_conductor_voltage_slider,
                user_data=idx,
                parent=self.conductor_controls_container_id,
            )
            self.conductor_slider_ids[idx] = slider_id

            # Add blur slider with fractional toggle
            with dpg.group(horizontal=True, parent=self.conductor_controls_container_id):
                if conductor.blur_is_fractional:
                    max_val = 0.1
                    fmt = "%.3f"
                else:
                    max_val = 20.0
                    fmt = "%.1f px"
                dpg.add_slider_float(
                    label=f"  Blur {idx + 1}",
                    default_value=float(conductor.blur_sigma),
                    min_value=0.0,
                    max_value=max_val,
                    format=fmt,
                    callback=self._on_conductor_blur_slider,
                    user_data=idx,
                    width=200,
                    tag=f"blur_slider_{idx}",
                )
                dpg.add_checkbox(
                    label="Frac",
                    default_value=conductor.blur_is_fractional,
                    callback=self._on_blur_fractional_toggle,
                    user_data=idx,
                    tag=f"blur_frac_checkbox_{idx}",
                )


    def _update_conductor_slider_labels(self, skip_idx: Optional[int] = None) -> None:
        if dpg is None or not self.conductor_slider_ids:
            return

        with self.state_lock:
            conductors = list(self.state.project.conductors)
            selected_idx = self.state.selected_idx

        for idx, slider_id in list(self.conductor_slider_ids.items()):
            if slider_id is None or not dpg.does_item_exist(slider_id):
                continue
            if idx >= len(conductors):
                continue
            label = f"C{idx + 1}"
            if idx == selected_idx:
                label += " (selected)"
            dpg.configure_item(slider_id, label=label)
            if skip_idx is not None and idx == skip_idx:
                continue
            dpg.set_value(slider_id, float(conductors[idx].voltage))

    def _update_region_properties_panel(self) -> None:
        """Update region properties panel based on current selection."""
        if dpg is None:
            return

        with self.state_lock:
            selected = self.state.get_selected()
            if selected and selected.id is not None:
                settings = self.state.conductor_color_settings.get(selected.id)
                if settings:
                    # Update checkboxes only (colormap buttons are always visible)
                    dpg.set_value("surface_enabled_checkbox", settings.surface.enabled)
                    dpg.set_value("interior_enabled_checkbox", settings.interior.enabled)

                # Update smear controls
                dpg.set_value("smear_enabled_checkbox", selected.smear_enabled)
                dpg.set_value("smear_sigma_slider", selected.smear_sigma)
                # Show/hide slider based on smear enabled
                dpg.configure_item("smear_sigma_slider", show=selected.smear_enabled)

    def _on_conductor_voltage_slider(self, sender, app_data, user_data):
        if dpg is None:
            return
        idx = int(user_data)
        value = float(app_data)
        with self.state_lock:
            actions.set_conductor_voltage(self.state, idx, value)
        self._mark_canvas_dirty()
        dpg.set_value("status_text", f"C{idx + 1} voltage = {value:.3f}")
        self._update_conductor_slider_labels(skip_idx=idx)

    def _on_conductor_blur_slider(self, sender, app_data, user_data):
        if dpg is None:
            return
        idx = int(user_data)
        value = float(app_data)
        with self.state_lock:
            if idx < len(self.state.project.conductors):
                self.state.project.conductors[idx].blur_sigma = value
                self.state.field_cache = None
                is_frac = self.state.project.conductors[idx].blur_is_fractional
        self._mark_canvas_dirty()
        if is_frac:
            dpg.set_value("status_text", f"C{idx + 1} blur = {value:.3f} (fraction)")
        else:
            dpg.set_value("status_text", f"C{idx + 1} blur = {value:.1f} px")

    def _on_blur_fractional_toggle(self, sender, app_data, user_data):
        if dpg is None:
            return
        idx = int(user_data)
        is_fractional = bool(app_data)
        with self.state_lock:
            if idx < len(self.state.project.conductors):
                conductor = self.state.project.conductors[idx]
                conductor.blur_is_fractional = is_fractional
                # Convert value when switching modes
                if is_fractional:
                    # Convert from pixels to fraction (assume ~1000px reference)
                    conductor.blur_sigma = min(conductor.blur_sigma / 1000.0, 0.1)
                else:
                    # Convert from fraction to pixels
                    conductor.blur_sigma = conductor.blur_sigma * 1000.0
                self.state.field_cache = None
        # Update slider range and format
        slider_id = f"blur_slider_{idx}"
        if dpg.does_item_exist(slider_id):
            if is_fractional:
                dpg.configure_item(slider_id, max_value=0.1, format="%.3f")
            else:
                dpg.configure_item(slider_id, max_value=20.0, format="%.1f px")
            with self.state_lock:
                if idx < len(self.state.project.conductors):
                    dpg.set_value(slider_id, self.state.project.conductors[idx].blur_sigma)
        self._mark_canvas_dirty()

    def _on_smear_enabled(self, sender, app_data):
        """Toggle interior smear for selected conductor region."""
        if dpg is None:
            return
        with self.state_lock:
            idx, region = self.state.selected_idx, self.selected_region
            if idx >= 0 and idx < len(self.state.project.conductors):
                self.state.project.conductors[idx].smear_enabled = bool(app_data)
        self._update_region_properties_panel()
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_smear_sigma(self, sender, app_data):
        """Adjust smear blur sigma for selected conductor."""
        if dpg is None:
            return
        with self.state_lock:
            idx = self.state.selected_idx
            if idx >= 0 and idx < len(self.state.project.conductors):
                self.state.project.conductors[idx].smear_sigma = float(app_data)
        self._refresh_render_texture()
        self._mark_canvas_dirty()

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
        self._update_conductor_slider_labels()
        self._update_region_properties_panel()
        dpg.set_value("status_text", f"Scaled C{idx + 1} by {factor:.2f}×")
        return True

    def _ensure_render_modal(self) -> None:
        if dpg is None or self.render_modal_id is not None:
            return

        with dpg.window(
            label="Render Settings",
            modal=True,
            show=False,
            tag="render_modal",
            no_move=False,
            no_close=True,
            no_collapse=True,
            width=420,
            height=520,
        ) as modal:
            self.render_modal_id = modal

            dpg.add_text("Supersample Factor")
            self.render_supersample_radio_id = dpg.add_radio_button(
                SUPERSAMPLE_LABELS,
                horizontal=True,
            )
            dpg.add_spacer(height=10)

            dpg.add_text("Render Resolution")
            self.render_multiplier_radio_id = dpg.add_radio_button(
                RESOLUTION_LABELS,
                horizontal=True,
            )
            dpg.add_spacer(height=12)

            dpg.add_separator()
            dpg.add_spacer(height=12)

            self.render_passes_input_id = dpg.add_input_int(
                label="LIC Passes",
                min_value=1,
                step=1,
                min_clamped=True,
                width=160,
            )

            self.render_streamlength_input_id = dpg.add_input_float(
                label="Streamlength Factor",
                format="%.4f",
                min_value=1e-6,
                min_clamped=True,
                step=0.0,
                width=200,
            )

            self.render_margin_input_id = dpg.add_input_float(
                label="Padding Margin",
                format="%.3f",
                min_value=0.0,
                min_clamped=True,
                step=0.0,
                width=200,
            )

            self.render_seed_input_id = dpg.add_input_int(
                label="Noise Seed",
                step=1,
                min_clamped=False,
                width=160,
            )

            self.render_sigma_input_id = dpg.add_input_float(
                label="Noise Low-pass Sigma",
                format="%.2f",
                min_value=0.0,
                min_clamped=True,
                step=0.0,
                width=200,
            )

            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            dpg.add_text("Boundary Conditions")
            dpg.add_spacer(height=5)

            # Cross-shaped layout for boundary controls
            with dpg.table(header_row=False, borders_innerH=False, borders_innerV=False,
                          borders_outerH=False, borders_outerV=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=100)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=120)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=100)

                # Row 0: Top boundary centered
                with dpg.table_row():
                    dpg.add_text("")
                    self.boundary_top_checkbox_id = dpg.add_checkbox(label="Top Neumann")
                    dpg.add_text("")

                # Row 1: Left and Right boundaries
                with dpg.table_row():
                    self.boundary_left_checkbox_id = dpg.add_checkbox(label="Left Neumann")
                    dpg.add_text("(Insulating)", indent=30)
                    self.boundary_right_checkbox_id = dpg.add_checkbox(label="Right Neumann")

                # Row 2: Bottom boundary centered
                with dpg.table_row():
                    dpg.add_text("")
                    self.boundary_bottom_checkbox_id = dpg.add_checkbox(label="Bottom Neumann")
                    dpg.add_text("")

            dpg.add_spacer(height=20)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Render", width=140, callback=self._apply_render_modal)
                dpg.add_button(label="Cancel", width=140, callback=self._cancel_render_modal)

    def _ensure_conductor_file_dialog(self) -> None:
        if dpg is None or self.conductor_file_dialog_id is not None:
            return

        # Default to assets/masks if it exists, otherwise use cwd
        masks_path = Path.cwd() / "assets" / "masks"
        default_path = str(masks_path) if masks_path.exists() else str(Path.cwd())

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            default_path=default_path,
            callback=self._on_conductor_file_selected,
            cancel_callback=self._on_conductor_file_cancelled,
            width=640,
            height=420,
            tag="conductor_file_dialog",
        ) as dialog:
            self.conductor_file_dialog_id = dialog
            dpg.add_file_extension(".png", color=(150, 180, 255, 255))
            dpg.add_file_extension(".*")

    def _open_render_modal(self, sender, app_data):
        if dpg is None:
            return
        if self.render_future is not None and not self.render_future.done():
            dpg.set_value("status_text", "Render already in progress...")
            return
        self._ensure_render_modal()
        self._update_render_modal_values()
        if self.render_modal_id is not None:
            dpg.configure_item(self.render_modal_id, show=True)
            self.render_modal_open = True

    def _close_render_modal(self) -> None:
        if dpg is None or self.render_modal_id is None:
            return
        dpg.configure_item(self.render_modal_id, show=False)
        self.render_modal_open = False

    def _open_conductor_dialog(self, sender, app_data):
        if dpg is None:
            return
        with self.state_lock:
            if self.state.view_mode != "edit":
                dpg.set_value("status_text", "Switch to edit mode to add conductors.")
                return

        self._ensure_conductor_file_dialog()
        if self.conductor_file_dialog_id is not None:
            dpg.show_item(self.conductor_file_dialog_id)

    def _on_conductor_file_cancelled(self, sender, app_data):
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)
        dpg.set_value("status_text", "Load conductor cancelled.")

    def _on_conductor_file_selected(self, sender, app_data):
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)

        path_str: Optional[str] = None
        if isinstance(app_data, dict):
            # Try selections dict FIRST - it's more reliable than file_path_name
            # (DPG has a bug where file_path_name becomes ".png" on second use)
            selections = app_data.get("selections", {})
            if selections:
                path_str = next(iter(selections.values()))

            if not path_str:
                # Fallback: try file_path_name
                path_str = app_data.get("file_path_name")

            if not path_str:
                # Fallback: combine current_path + file_name
                current_path = app_data.get("current_path", "")
                file_name = app_data.get("file_name", "")
                if current_path and file_name:
                    path_str = str(Path(current_path) / file_name)

        if not path_str:
            dpg.set_value("status_text", "No file selected.")
            return

        # Convert to absolute path to handle relative paths
        try:
            path_obj = Path(path_str)
            if not path_obj.is_absolute():
                path_obj = path_obj.resolve()
            path_str = str(path_obj)
        except Exception:
            pass  # If path resolution fails, try with original path_str

        try:
            mask, interior = load_conductor_masks(path_str)
        except Exception as exc:  # pragma: no cover - PIL errors etc.
            dpg.set_value("status_text", f"Failed to load conductor: {exc}")
            return

        mask_h, mask_w = mask.shape
        if mask_w > MAX_CANVAS_DIM or mask_h > MAX_CANVAS_DIM:
            dpg.set_value("status_text", f"Mask exceeds max dimension {MAX_CANVAS_DIM}px.")
            return

        with self.state_lock:
            project = self.state.project
            if len(project.conductors) == 0:
                canvas_w, canvas_h = project.canvas_resolution
                new_w = max(canvas_w, mask_w)
                new_h = max(canvas_h, mask_h)
                if (new_w, new_h) != project.canvas_resolution:
                    actions.set_canvas_resolution(self.state, new_w, new_h)

            canvas_w, canvas_h = self.state.project.canvas_resolution
            # Offset each new conductor by 30px down-right from center so they're all visible
            num_conductors = len(project.conductors)
            offset = num_conductors * 30.0
            pos = ((canvas_w - mask_w) / 2.0 + offset, (canvas_h - mask_h) / 2.0 + offset)
            conductor = Conductor(mask=mask, voltage=0.5, position=pos, interior_mask=interior)
            actions.add_conductor(self.state, conductor)
            self.state.view_mode = "edit"

        self._mark_canvas_dirty()
        self._update_control_visibility()
        self._rebuild_conductor_controls()
        self._update_conductor_slider_labels()
        dpg.set_value("status_text", f"Loaded conductor '{Path(path_str).name}'")

    # ------------------------------------------------------------------
    # Project save/load
    # ------------------------------------------------------------------
    def _ensure_save_project_dialog(self) -> None:
        if dpg is None or self.save_project_dialog_id is not None:
            return

        # Default to projects/ directory
        projects_path = Path.cwd() / "projects"
        default_path = str(projects_path) if projects_path.exists() else str(Path.cwd())

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            default_path=default_path,
            callback=self._on_save_project_file_selected,
            cancel_callback=self._on_save_project_cancelled,
            width=640,
            height=420,
            tag="save_project_dialog",
            default_filename="project.flowcol",
        ) as dialog:
            self.save_project_dialog_id = dialog
            dpg.add_file_extension(".flowcol", color=(180, 255, 150, 255))
            dpg.add_file_extension(".*")

    def _ensure_load_project_dialog(self) -> None:
        if dpg is None or self.load_project_dialog_id is not None:
            return

        # Default to projects/ directory
        projects_path = Path.cwd() / "projects"
        default_path = str(projects_path) if projects_path.exists() else str(Path.cwd())

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            default_path=default_path,
            callback=self._on_load_project_file_selected,
            cancel_callback=self._on_load_project_cancelled,
            width=640,
            height=420,
            tag="load_project_dialog",
        ) as dialog:
            self.load_project_dialog_id = dialog
            dpg.add_file_extension(".flowcol", color=(180, 255, 150, 255))
            dpg.add_file_extension(".*")

    def _open_save_project_dialog(self, sender, app_data):
        if dpg is None:
            return
        self._ensure_save_project_dialog()
        if self.save_project_dialog_id is not None:
            dpg.show_item(self.save_project_dialog_id)

    def _open_load_project_dialog(self, sender, app_data):
        if dpg is None:
            return
        self._ensure_load_project_dialog()
        if self.load_project_dialog_id is not None:
            dpg.show_item(self.load_project_dialog_id)

    def _on_save_project_cancelled(self, sender, app_data):
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)
        dpg.set_value("status_text", "Save project cancelled.")

    def _on_load_project_cancelled(self, sender, app_data):
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)
        dpg.set_value("status_text", "Load project cancelled.")

    def _on_save_project_file_selected(self, sender, app_data):
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)

        path_str = self._extract_file_path(app_data)
        if not path_str:
            dpg.set_value("status_text", "No file selected.")
            return

        # Ensure .flowcol extension
        path_obj = Path(path_str)
        if path_obj.suffix != '.flowcol':
            path_obj = path_obj.with_suffix('.flowcol')

        try:
            with self.state_lock:
                save_project(self.state, str(path_obj))

                # Track current project path
                self.current_project_path = str(path_obj)

                # Also save render cache if it exists
                if self.state.render_cache is not None:
                    cache_path = path_obj.with_suffix('.flowcol.cache')
                    save_render_cache(self.state.render_cache, self.state.project, str(cache_path))
                    cache_size_mb = cache_path.stat().st_size / 1024 / 1024
                    dpg.set_value("status_text", f"Saved project + cache ({cache_size_mb:.1f} MB): {path_obj.name}")
                else:
                    dpg.set_value("status_text", f"Saved project: {path_obj.name}")
        except Exception as exc:
            dpg.set_value("status_text", f"Failed to save project: {exc}")

    def _on_load_project_file_selected(self, sender, app_data):
        if dpg is None:
            return
        if sender is not None:
            dpg.configure_item(sender, show=False)

        path_str = self._extract_file_path(app_data)
        if not path_str:
            dpg.set_value("status_text", "No file selected.")
            return

        try:
            new_state = load_project(path_str)

            # Try to load render cache
            cache_path = Path(path_str).with_suffix('.flowcol.cache')
            loaded_cache = load_render_cache(str(cache_path), new_state.project)

            with self.state_lock:
                # Replace current state with loaded state
                self.state.project = new_state.project
                self.state.render_settings = new_state.render_settings
                self.state.display_settings = new_state.display_settings
                self.state.conductor_color_settings = new_state.conductor_color_settings
                self.state.selected_idx = -1
                self.state.view_mode = "edit"
                self.state.field_dirty = True
                self.state.render_dirty = True
                self.state.render_cache = loaded_cache

            # Track current project path
            self.current_project_path = path_str

            # Rebuild display fields from loaded cache
            if loaded_cache is not None:
                self._rebuild_cache_display_fields()

            # Update UI to reflect loaded state
            self._mark_canvas_dirty()
            self._update_canvas_scale()  # Recalculate scale for new canvas resolution
            self._update_control_visibility()
            self._rebuild_conductor_controls()
            self._update_conductor_slider_labels()
            self._sync_ui_from_state()
            self._update_cache_status_display()

            # Status message
            if loaded_cache:
                shape = loaded_cache.result.array.shape
                dpg.set_value("status_text", f"Loaded project with render cache ({shape[1]}×{shape[0]})")
            else:
                dpg.set_value("status_text", f"Loaded project: {Path(path_str).name}")
        except Exception as exc:
            dpg.set_value("status_text", f"Failed to load project: {exc}")

    def _extract_file_path(self, app_data) -> Optional[str]:
        """Extract file path from DPG file dialog callback data."""
        if not isinstance(app_data, dict):
            return None

        # Try selections dict first
        selections = app_data.get("selections", {})
        if selections:
            return next(iter(selections.values()))

        # Fallback: file_path_name
        path_str = app_data.get("file_path_name")
        if path_str:
            return path_str

        # Fallback: combine current_path + file_name
        current_path = app_data.get("current_path", "")
        file_name = app_data.get("file_name", "")
        if current_path and file_name:
            return str(Path(current_path) / file_name)

        return None

    def _sync_ui_from_state(self) -> None:
        """Sync UI controls to match current state after loading."""
        if dpg is None:
            return

        with self.state_lock:
            # Canvas resolution
            if self.canvas_width_input_id is not None:
                dpg.set_value(self.canvas_width_input_id, self.state.project.canvas_resolution[0])
            if self.canvas_height_input_id is not None:
                dpg.set_value(self.canvas_height_input_id, self.state.project.canvas_resolution[1])

            # Display settings
            if self.postprocess_downsample_slider_id is not None:
                dpg.set_value(self.postprocess_downsample_slider_id, self.state.display_settings.downsample_sigma)
            if self.postprocess_clip_slider_id is not None:
                dpg.set_value(self.postprocess_clip_slider_id, self.state.display_settings.clip_percent)
            if self.postprocess_brightness_slider_id is not None:
                dpg.set_value(self.postprocess_brightness_slider_id, self.state.display_settings.brightness)
            if self.postprocess_contrast_slider_id is not None:
                dpg.set_value(self.postprocess_contrast_slider_id, self.state.display_settings.contrast)
            if self.postprocess_gamma_slider_id is not None:
                dpg.set_value(self.postprocess_gamma_slider_id, self.state.display_settings.gamma)
            if self.color_enabled_checkbox_id is not None:
                dpg.set_value(self.color_enabled_checkbox_id, self.state.display_settings.color_enabled)

    # ------------------------------------------------------------------
    # Render Cache Management
    # ------------------------------------------------------------------
    def _update_cache_status_display(self) -> None:
        """Update cache status text and warning visibility."""
        if dpg is None:
            return

        with self.state_lock:
            cache = self.state.render_cache

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
            current_fp = compute_project_fingerprint(self.state.project)
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

    def _on_mark_clean_clicked(self, sender, app_data):
        """Mark cache as clean (reset fingerprint to current project state)."""
        with self.state_lock:
            if self.state.render_cache is not None:
                current_fp = compute_project_fingerprint(self.state.project)
                self.state.render_cache.project_fingerprint = current_fp

        self._update_cache_status_display()
        dpg.set_value("status_text", "Render cache marked as clean")

    def _on_discard_cache_clicked(self, sender, app_data):
        """Discard cached render."""
        with self.state_lock:
            self.state.render_cache = None
            self.state.view_mode = "edit"

        self._update_control_visibility()
        self._mark_canvas_dirty()
        self._update_cache_status_display()
        dpg.set_value("status_text", "Render cache discarded")

    def _on_view_postprocessing_clicked(self, sender, app_data):
        """Switch to render mode using existing cache (no re-render)."""
        with self.state_lock:
            if self.state.render_cache is None:
                dpg.set_value("status_text", "No cached render available")
                return
            self.state.view_mode = "render"

        self._update_control_visibility()
        self._mark_canvas_dirty()
        self._refresh_render_texture()
        dpg.set_value("status_text", "Viewing cached render")

    def _auto_save_cache(self) -> None:
        """Auto-save render cache to disk (called after successful render)."""
        if self.current_project_path is None:
            return

        try:
            cache_path = Path(self.current_project_path).with_suffix('.flowcol.cache')
            with self.state_lock:
                if self.state.render_cache is not None:
                    save_render_cache(self.state.render_cache, self.state.project, str(cache_path))
                    cache_size_mb = cache_path.stat().st_size / 1024 / 1024
                    dpg.set_value("status_text", f"Render complete. Cache saved ({cache_size_mb:.1f} MB)")
        except Exception as exc:
            dpg.set_value("status_text", f"Render complete. Failed to save cache: {exc}")

    def _rebuild_cache_display_fields(self) -> None:
        """Rebuild display_array and masks from loaded cache RenderResult."""
        from flowcol.render import downsample_lic
        from flowcol.postprocess.masks import rasterize_conductor_masks
        from scipy.ndimage import zoom

        with self.state_lock:
            cache = self.state.render_cache
            if cache is None or cache.result is None:
                return

            # Recompute display_array
            canvas_w, canvas_h = self.state.project.canvas_resolution
            target_shape = (canvas_h, canvas_w)
            display_array = downsample_lic(
                cache.result.array,
                target_shape,
                cache.supersample,
                self.state.display_settings.downsample_sigma,
            )
            # Set CPU as primary source (this is CPU-only path)
            cache.set_display_array_cpu(display_array)

            # Use cached masks from RenderResult if available (avoids redundant rasterization)
            if self.state.project.conductors:
                if cache.result.conductor_masks_canvas is not None:
                    # Use pre-computed masks from render
                    full_res_conductor_masks = cache.result.conductor_masks_canvas
                    full_res_interior_masks = cache.result.interior_masks_canvas
                else:
                    # Fallback: rasterize masks (for compatibility with older cached renders)
                    scale = cache.multiplier * cache.supersample
                    full_res_conductor_masks, full_res_interior_masks = rasterize_conductor_masks(
                        self.state.project.conductors,
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


    def _update_render_modal_values(self) -> None:
        if dpg is None:
            return
        with self.state_lock:
            settings = replace(self.state.render_settings)
            streamlength = self.state.project.streamlength_factor
            project = self.state.project

        if self.render_supersample_radio_id is not None:
            dpg.set_value(self.render_supersample_radio_id, _label_for_supersample(settings.supersample))

        if self.render_multiplier_radio_id is not None:
            dpg.set_value(self.render_multiplier_radio_id, _label_for_multiplier(settings.multiplier))

        if self.render_passes_input_id is not None:
            dpg.set_value(self.render_passes_input_id, int(settings.num_passes))

        if self.render_streamlength_input_id is not None:
            dpg.set_value(self.render_streamlength_input_id, float(streamlength))

        if self.render_margin_input_id is not None:
            dpg.set_value(self.render_margin_input_id, float(settings.margin))

        if self.render_seed_input_id is not None:
            dpg.set_value(self.render_seed_input_id, int(settings.noise_seed))

        if self.render_sigma_input_id is not None:
            dpg.set_value(self.render_sigma_input_id, float(settings.noise_sigma))

        # Update boundary condition checkboxes
        from flowcol.poisson import NEUMANN
        if self.boundary_top_checkbox_id is not None:
            dpg.set_value(self.boundary_top_checkbox_id, project.boundary_top == NEUMANN)
        if self.boundary_bottom_checkbox_id is not None:
            dpg.set_value(self.boundary_bottom_checkbox_id, project.boundary_bottom == NEUMANN)
        if self.boundary_left_checkbox_id is not None:
            dpg.set_value(self.boundary_left_checkbox_id, project.boundary_left == NEUMANN)
        if self.boundary_right_checkbox_id is not None:
            dpg.set_value(self.boundary_right_checkbox_id, project.boundary_right == NEUMANN)

    def _cancel_render_modal(self, sender=None, app_data=None) -> None:
        self._close_render_modal()

    def _apply_render_modal(self, sender=None, app_data=None) -> None:
        if dpg is None:
            return

        supersample_label = dpg.get_value(self.render_supersample_radio_id) if self.render_supersample_radio_id else SUPERSAMPLE_LABELS[0]
        multiplier_label = dpg.get_value(self.render_multiplier_radio_id) if self.render_multiplier_radio_id else RESOLUTION_LABELS[0]

        supersample = SUPERSAMPLE_LOOKUP.get(supersample_label, SUPERSAMPLE_CHOICES[0])
        multiplier = RESOLUTION_LOOKUP.get(multiplier_label, RESOLUTION_CHOICES[0])

        passes = int(dpg.get_value(self.render_passes_input_id)) if self.render_passes_input_id is not None else defaults.DEFAULT_RENDER_PASSES
        streamlength = float(dpg.get_value(self.render_streamlength_input_id)) if self.render_streamlength_input_id is not None else defaults.DEFAULT_STREAMLENGTH_FACTOR
        margin = float(dpg.get_value(self.render_margin_input_id)) if self.render_margin_input_id is not None else defaults.DEFAULT_PADDING_MARGIN
        noise_seed = int(dpg.get_value(self.render_seed_input_id)) if self.render_seed_input_id is not None else defaults.DEFAULT_NOISE_SEED
        noise_sigma = float(dpg.get_value(self.render_sigma_input_id)) if self.render_sigma_input_id is not None else defaults.DEFAULT_NOISE_SIGMA

        # Read boundary condition checkboxes
        from flowcol.poisson import DIRICHLET, NEUMANN
        boundary_top = NEUMANN if (self.boundary_top_checkbox_id and dpg.get_value(self.boundary_top_checkbox_id)) else DIRICHLET
        boundary_bottom = NEUMANN if (self.boundary_bottom_checkbox_id and dpg.get_value(self.boundary_bottom_checkbox_id)) else DIRICHLET
        boundary_left = NEUMANN if (self.boundary_left_checkbox_id and dpg.get_value(self.boundary_left_checkbox_id)) else DIRICHLET
        boundary_right = NEUMANN if (self.boundary_right_checkbox_id and dpg.get_value(self.boundary_right_checkbox_id)) else DIRICHLET

        # Clamp to valid ranges similar to pygame UI
        passes = max(passes, 1)
        streamlength = max(streamlength, 1e-6)
        margin = max(margin, 0.0)
        noise_sigma = max(noise_sigma, 0.0)

        with self.state_lock:
            actions.set_supersample(self.state, supersample)
            actions.set_render_multiplier(self.state, multiplier)
            actions.set_num_passes(self.state, passes)
            actions.set_margin(self.state, margin)
            actions.set_noise_seed(self.state, noise_seed)
            actions.set_noise_sigma(self.state, noise_sigma)
            actions.set_streamlength_factor(self.state, streamlength)
            # Update boundary conditions
            self.state.project.boundary_top = boundary_top
            self.state.project.boundary_bottom = boundary_bottom
            self.state.project.boundary_left = boundary_left
            self.state.project.boundary_right = boundary_right

        self._close_render_modal()
        self._start_render_job()

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
                self._update_conductor_slider_labels()
                self._update_region_properties_panel()
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
            self._update_conductor_slider_labels()
            self._update_region_properties_panel()
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
                self._rebuild_conductor_controls()
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
                    self._rebuild_conductor_controls()
                    self._update_conductor_slider_labels()
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

        # Colorize using optimized path (JIT-accelerated on CPU)
        from flowcol.postprocess.color import build_base_rgb
        base_rgb = build_base_rgb(lic_array, settings.to_color_params(), display_array_gpu=None)

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

        # Apply smear
        if any(c.smear_enabled for c in project.conductors):
            # Compute percentiles for this resolution (export can be at any resolution)
            vmin = float(np.percentile(lic_array, 0.5))
            vmax = float(np.percentile(lic_array, 99.5))
            lic_percentiles = (vmin, vmax)

            base_rgb = apply_conductor_smear(
                base_rgb,
                lic_array,
                project,
                settings.palette,
                lic_array.shape,
                color_enabled=settings.color_enabled,
                lic_percentiles=lic_percentiles,
            )

        # Apply region overlays
        if conductor_masks and interior_masks:
            final_rgb = apply_region_overlays(
                base_rgb,
                lic_array,
                conductor_masks,
                interior_masks,
                conductor_color_settings,
                project.conductors,
                settings.to_color_params(),
            )
        else:
            final_rgb = base_rgb

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

    def _on_downsample_slider(self, sender, app_data):
        """Handle downsampling blur sigma slider change (real-time with GPU acceleration)."""
        value = float(app_data)
        with self.state_lock:
            self.state.display_settings.downsample_sigma = value

        # GPU is fast enough for real-time updates - no debouncing needed!
        self._apply_postprocessing()

    def _on_clip_slider(self, sender, app_data):
        """Handle clip percent slider change."""
        value = float(app_data)
        with self.state_lock:
            self.state.display_settings.clip_percent = value
            self.state.invalidate_base_rgb()
        # Clip is display-only, just refresh texture
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_brightness_slider(self, sender, app_data):
        """Handle brightness slider change (real-time with GPU acceleration)."""
        value = float(app_data)
        with self.state_lock:
            self.state.display_settings.brightness = value
            self.state.invalidate_base_rgb()

        # GPU is fast enough for real-time updates - no debouncing needed!
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_contrast_slider(self, sender, app_data):
        """Handle contrast slider change (real-time with GPU acceleration)."""
        value = float(app_data)
        with self.state_lock:
            self.state.display_settings.contrast = value
            self.state.invalidate_base_rgb()

        # GPU is fast enough for real-time updates - no debouncing needed!
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_gamma_slider(self, sender, app_data):
        """Handle gamma slider change (real-time with GPU acceleration)."""
        value = float(app_data)
        with self.state_lock:
            self.state.display_settings.gamma = value
            self.state.invalidate_base_rgb()

        # GPU is fast enough for real-time updates - no debouncing needed!
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_color_enabled(self, sender, app_data):
        """Handle color enabled checkbox change."""
        from flowcol.app.actions import set_color_enabled
        with self.state_lock:
            set_color_enabled(self.state, app_data)
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_palette_changed(self, sender, app_data):
        """Handle palette dropdown change."""
        from flowcol.app.actions import set_palette
        with self.state_lock:
            set_palette(self.state, app_data)
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_global_palette_button(self, sender, app_data, user_data):
        """Handle global colormap button click."""
        from flowcol.app.actions import set_palette
        palette_name = user_data
        with self.state_lock:
            set_palette(self.state, palette_name)
        # Update current palette display
        if dpg is not None:
            dpg.set_value("global_palette_current_text", f"Current: {palette_name}")
            dpg.configure_item("global_palette_popup", show=False)
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_surface_enabled(self, sender, app_data):
        """Handle surface custom palette checkbox."""
        from flowcol.app.actions import set_region_style_enabled
        with self.state_lock:
            selected = self.state.get_selected()
            if selected and selected.id is not None:
                set_region_style_enabled(self.state, selected.id, "surface", app_data)
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_surface_palette_button(self, sender, app_data, user_data):
        """Handle surface colormap button click."""
        from flowcol.app.actions import set_region_palette
        palette_name = user_data
        with self.state_lock:
            selected = self.state.get_selected()
            if selected and selected.id is not None:
                set_region_palette(self.state, selected.id, "surface", palette_name)
        # Update current palette display
        if dpg is not None:
            dpg.set_value("surface_palette_current_text", f"Current: {palette_name}")
            dpg.configure_item("surface_palette_popup", show=False)
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_interior_enabled(self, sender, app_data):
        """Handle interior custom color checkbox."""
        from flowcol.app.actions import set_region_style_enabled
        with self.state_lock:
            selected = self.state.get_selected()
            if selected and selected.id is not None:
                set_region_style_enabled(self.state, selected.id, "interior", app_data)
        self._refresh_render_texture()
        self._mark_canvas_dirty()

    def _on_interior_palette_button(self, sender, app_data, user_data):
        """Handle interior colormap button click."""
        from flowcol.app.actions import set_region_palette
        palette_name = user_data
        with self.state_lock:
            selected = self.state.get_selected()
            if selected and selected.id is not None:
                set_region_palette(self.state, selected.id, "interior", palette_name)
        # Update current palette display
        if dpg is not None:
            dpg.set_value("interior_palette_current_text", f"Current: {palette_name}")
            dpg.configure_item("interior_palette_popup", show=False)
        self._refresh_render_texture()
        self._mark_canvas_dirty()

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
    # Rendering
    # ------------------------------------------------------------------
    def _start_render_job(self) -> None:
        if self.render_future is not None and not self.render_future.done():
            return

        self.render_error = None
        dpg.set_value("status_text", "Rendering...")

        def job() -> bool:
            with self.state_lock:
                settings_snapshot: RenderSettings = replace(self.state.render_settings)
                project_snapshot = _snapshot_project(self.state.project)
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
            from flowcol.postprocess.masks import rasterize_conductor_masks

            with self.state_lock:
                postprocess = self.state.display_settings
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

            with self.state_lock:
                self.state.render_cache = cache
                self.state.field_dirty = False
                self.state.render_dirty = False
                self.state.view_mode = "render"

            # Auto-save high-res renders (>2k on any dimension)
            render_h, render_w = result.array.shape
            if render_w >= 2000 or render_h >= 2000:
                from PIL import Image
                from datetime import datetime
                from flowcol.render import downsample_lic

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

        self.render_future = self.executor.submit(job)

    def _poll_render_future(self) -> None:
        if self.render_future is None:
            return
        if not self.render_future.done():
            return

        success = False
        try:
            success = bool(self.render_future.result())
        except Exception as exc:  # pragma: no cover - unexpected
            success = False
            self.render_error = str(exc)

        self.render_future = None

        if success:
            self._mark_canvas_dirty()
            self._refresh_render_texture()
            self.drag_active = False
            with self.state_lock:
                self.state.view_mode = "render"
            self._update_control_visibility()
            self._update_cache_status_display()

            # Auto-save cache if project has been saved before
            if self.current_project_path is not None:
                self._auto_save_cache()
            else:
                dpg.set_value("status_text", "Render complete. (Save project to preserve cache)")
        else:
            msg = self.render_error or "Render failed (possibly due to excessive resolution)."
            dpg.set_value("status_text", msg)

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
                self._poll_render_future()
                if self.canvas_dirty:
                    self._redraw_canvas()
                dpg.render_dearpygui_frame()
        finally:
            self.executor.shutdown(wait=False)
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

        if self.view_postprocessing_button_id is not None:
            show_button = (mode == "edit" and has_cache)
            dpg.configure_item(self.view_postprocessing_button_id, show=show_button)

        self._update_conductor_slider_labels()

def run() -> None:
    """Launch FlowCol Dear PyGui application."""
    FlowColApp().render()
