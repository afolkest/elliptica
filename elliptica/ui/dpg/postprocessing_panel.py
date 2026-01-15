"""Postprocessing panel controller for Elliptica UI - sliders, color, and region properties."""

import time
import numpy as np
from PIL import Image
from typing import Optional, Literal, TYPE_CHECKING

from elliptica import defaults
from elliptica.app import actions
from elliptica.app.core import resolve_region_postprocess_params
from elliptica.colorspace import (
    build_oklch_lut,
    gamut_map_to_srgb,
    interpolate_oklch,
    max_chroma_fast,
    srgb_to_oklch,
)
from elliptica.render import (
    delete_palette,
    get_palette_spec,
    list_color_palettes,
    set_palette_spec,
)

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class PostprocessingPanel:
    """Controller for postprocessing sliders, color controls, and region properties."""

    def __init__(self, app: "EllipticaApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main EllipticaApp instance
        """
        self.app = app

        # Widget IDs for postprocessing sliders
        self.postprocess_clip_low_slider_id: Optional[int] = None
        self.postprocess_clip_high_slider_id: Optional[int] = None
        self.postprocess_brightness_slider_id: Optional[int] = None
        self.postprocess_contrast_slider_id: Optional[int] = None
        self.postprocess_gamma_slider_id: Optional[int] = None

        # Widget IDs for smear controls
        self.smear_enabled_checkbox_id: Optional[int] = None
        self.smear_sigma_slider_id: Optional[int] = None

        # Widget IDs for expression editor
        self.expr_L_input_id: Optional[int] = None
        self.expr_C_input_id: Optional[int] = None
        self.expr_H_input_id: Optional[int] = None
        self.expr_error_text_id: Optional[int] = None
        self.expr_preset_combo_id: Optional[int] = None

        # Widget IDs for lightness expression (palette mode)
        self.lightness_expr_checkbox_id: Optional[int] = None
        self.lightness_expr_input_id: Optional[int] = None

        # Palette preview + histogram UI
        self.palette_preview_width = 320
        self.palette_preview_height = 22
        self.palette_hist_height = 60
        self.palette_hist_bins = 128
        self.hist_max_samples = 1_000_000
        self.hist_target_shape = (512, 512)
        self.global_palette_preview_button_id: Optional[int] = None
        self.region_palette_preview_button_id: Optional[int] = None
        self.global_hist_drawlist_id: Optional[int] = None
        self.region_hist_drawlist_id: Optional[int] = None
        self.global_hist_values: Optional[np.ndarray] = None
        self.region_hist_values: Optional[np.ndarray] = None
        self.hist_pending_update: bool = False
        self.hist_last_update_time: float = 0.0
        self.hist_debounce_delay: float = 0.05  # 50ms throttle

        # Palette editor UI (shell for phase 3)
        self.palette_editor_active: bool = False
        self.palette_editor_palette_name: Optional[str] = None
        self.palette_editor_for_region: bool = False
        self.palette_editor_group_id: Optional[int] = None
        self.palette_editor_title_id: Optional[int] = None
        self.palette_editor_done_button_id: Optional[int] = None
        self.palette_editor_gradient_drawlist_id: Optional[int] = None
        self.palette_editor_slice_drawlist_id: Optional[int] = None
        self.palette_editor_l_gradient_drawlist_id: Optional[int] = None
        self.palette_editor_interp_mode_id: Optional[int] = None
        self.palette_editor_l_slider_id: Optional[int] = None
        self.palette_editor_c_slider_id: Optional[int] = None
        self.palette_editor_h_slider_id: Optional[int] = None

        self.palette_editor_width = self.palette_preview_width
        self.palette_editor_gradient_height = 60
        self.palette_editor_slice_height = 110
        self.palette_editor_lbar_height = 16
        self.palette_editor_lbar_width = self.palette_editor_width
        self.palette_editor_gradient_texture_id: Optional[int] = None
        self.palette_editor_slice_texture_id: Optional[int] = None
        self.palette_editor_l_gradient_texture_id: Optional[int] = None
        self.palette_editor_handler_registry_id: Optional[int] = None
        self.palette_editor_stops: list[dict] = []
        self.palette_editor_next_stop_id: int = 0
        self.palette_editor_selected_stop_id: Optional[int] = None
        self.palette_editor_dragging_stop_id: Optional[int] = None
        self.palette_editor_is_dragging: bool = False
        self.palette_editor_slice_drag_active: bool = False
        self.palette_editor_relative_chroma: bool = True
        self.palette_editor_interp_mix: float = 1.0
        self.palette_editor_c_max_absolute: float = 0.5
        self.palette_editor_c_max_display: float = 1.0
        self.palette_editor_max_stops: int = 12
        self.palette_editor_gradient_padding = 14
        self.palette_editor_gradient_bar_top = 6
        self.palette_editor_gradient_bar_bottom = self.palette_editor_gradient_height - 18
        self.palette_editor_handle_radius = 7
        self.palette_editor_handle_hit_radius = 12
        self.palette_editor_handle_center_y = self.palette_editor_gradient_bar_bottom
        self.palette_editor_h_grid = np.array([], dtype=np.float32)
        self.palette_editor_c_grid = np.array([], dtype=np.float32)
        self.palette_editor_h_mesh = np.array([], dtype=np.float32)
        self.palette_editor_c_mesh = np.array([], dtype=np.float32)
        self.palette_editor_refresh_pending: bool = False
        self.palette_editor_last_refresh_time: float = 0.0
        self.palette_editor_refresh_throttle: float = 0.1
        self.palette_editor_dirty: bool = False
        self.palette_editor_persist_dirty: bool = False
        self.palette_editor_needs_colormap_rebuild: bool = False
        self.palette_editor_syncing: bool = False
        self._rebuild_palette_editor_grids()

        # Color mode: "palette" or "expressions"
        self.color_mode: str = "palette"

        # Note: Region type is now stored in app.state.selected_region_type

        # Delete confirmation modal
        self.delete_confirmation_modal_id: Optional[int] = None
        self.pending_delete_palette: Optional[str] = None

        # Debouncing for expensive smear updates
        self.smear_pending_value: Optional[float] = None
        self.smear_last_update_time: float = 0.0
        self.smear_debounce_delay: float = 0.3  # 300ms delay

        # Debouncing for expensive clip updates (percentile computation at high res)
        self.clip_pending_range: Optional[tuple[float, float]] = None
        self.clip_last_update_time: float = 0.0
        self.clip_debounce_delay: float = 0.3  # 300ms delay

        # Debouncing for expression updates
        self.expr_pending_update: bool = False
        self.expr_last_update_time: float = 0.0
        self.expr_debounce_delay: float = 0.3  # 300ms delay

        # Debouncing for lightness expression updates
        self.lightness_expr_pending_update: bool = False
        self.lightness_expr_last_update_time: float = 0.0
        # Target for pending lightness expr update: "global", or (boundary_id, region_type) tuple
        self.lightness_expr_pending_target: str | tuple[int, str] | None = None

        # Cache for custom lightness expressions (preserved when switching to Global mode)
        # Key: (boundary_id, region_type), Value: expression string
        self._cached_custom_lightness_exprs: dict[tuple[int, str], str] = {}

        # Cache for global lightness expression (preserved when disabling)
        self._cached_global_lightness_expr: str | None = None

        # Themes for grayed-out appearance
        self.disabled_theme_id: Optional[int] = None
        self.disabled_slider_theme_id: Optional[int] = None
        self.disabled_button_theme_id: Optional[int] = None
        self.disabled_input_theme_id: Optional[int] = None

    def _ensure_themes(self) -> None:
        """Create disabled/normal themes if not already created."""
        if dpg is None or self.disabled_theme_id is not None:
            return

        # Grayed-out colors (obviously dimmed)
        gray_bg = (35, 35, 35, 255)
        gray_grab = (70, 70, 70, 255)
        gray_text = (90, 90, 90, 255)
        gray_button = (45, 45, 45, 255)

        # Disabled theme for sliders - uses enabled_state=False to apply when disabled
        with dpg.theme() as disabled_slider_theme:
            with dpg.theme_component(dpg.mvSliderFloat, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, gray_bg, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, gray_bg, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, gray_bg, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, gray_grab, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, gray_grab, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, gray_text, category=dpg.mvThemeCat_Core)
        self.disabled_slider_theme_id = disabled_slider_theme

        # Disabled theme for buttons
        with dpg.theme() as disabled_button_theme:
            with dpg.theme_component(dpg.mvButton, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_Button, gray_button, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, gray_button, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, gray_button, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, gray_text, category=dpg.mvThemeCat_Core)
        self.disabled_button_theme_id = disabled_button_theme

        # Disabled theme for input text
        with dpg.theme() as disabled_input_theme:
            with dpg.theme_component(dpg.mvInputText, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, gray_bg, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, gray_bg, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, gray_bg, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, gray_text, category=dpg.mvThemeCat_Core)
        self.disabled_input_theme_id = disabled_input_theme

        # Mark as initialized
        self.disabled_theme_id = disabled_slider_theme

    def _set_slider_grayed(self, widget_tag: str, grayed: bool) -> None:
        """Set a slider to visually grayed out or normal."""
        if dpg is None:
            return
        self._ensure_themes()
        if grayed:
            dpg.bind_item_theme(widget_tag, self.disabled_slider_theme_id)
            dpg.configure_item(widget_tag, enabled=False)
        else:
            dpg.bind_item_theme(widget_tag, 0)
            dpg.configure_item(widget_tag, enabled=True)

    def _set_button_grayed(self, widget_tag: str, grayed: bool) -> None:
        """Set a button to visually grayed out or normal."""
        if dpg is None:
            return
        self._ensure_themes()
        if grayed:
            dpg.bind_item_theme(widget_tag, self.disabled_button_theme_id)
            dpg.configure_item(widget_tag, enabled=False)
        else:
            dpg.bind_item_theme(widget_tag, 0)
            dpg.configure_item(widget_tag, enabled=True)

    def _set_input_grayed(self, widget_tag: str, grayed: bool) -> None:
        """Set an input text to visually grayed out or normal."""
        if dpg is None:
            return
        self._ensure_themes()
        if grayed:
            dpg.bind_item_theme(widget_tag, self.disabled_input_theme_id)
            dpg.configure_item(widget_tag, enabled=False)
        else:
            dpg.bind_item_theme(widget_tag, 0)
            dpg.configure_item(widget_tag, enabled=True)

    def _is_boundary_selected(self) -> bool:
        """Check if a boundary is currently selected."""
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            return selected is not None and selected.id is not None

    def _get_current_region_style_unlocked(self):
        """Get RegionStyle for current selection. Caller MUST hold state_lock."""
        selected = self.app.state.get_selected()
        if selected is None or selected.id is None:
            return None
        settings = self.app.state.boundary_color_settings.get(selected.id)
        if settings is None:
            return None
        if self.app.state.selected_region_type == "surface":
            return settings.surface
        else:
            return settings.interior

    def _get_current_region_style(self):
        """Get the RegionStyle for the currently selected context, or None if global.

        Returns a snapshot of key values, NOT a mutable reference.
        For mutable access, use _get_current_region_style_unlocked() while holding lock.
        """
        with self.app.state_lock:
            return self._get_current_region_style_unlocked()

    def _has_override_enabled(self) -> bool:
        """Check if the current region has override enabled (palette + sliders)."""
        with self.app.state_lock:
            region_style = self._get_current_region_style_unlocked()
            if region_style is None:
                return False
            return region_style.enabled

    def _get_slider_context(self) -> tuple[bool, int | None, str]:
        """Get all context needed for B/C/G slider callbacks in one lock acquisition.

        Returns:
            (has_override, boundary_id, region_type)
            - has_override: True if boundary selected AND region.enabled is True
            - boundary_id: Selected boundary's ID, or None
            - region_type: "surface" or "interior"
        """
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected is None or selected.id is None:
                return (False, None, "surface")

            settings = self.app.state.boundary_color_settings.get(selected.id)
            region_type = self.app.state.selected_region_type

            if settings is None:
                return (False, selected.id, region_type)

            region_style = settings.surface if region_type == "surface" else settings.interior
            return (region_style.enabled, selected.id, region_type)

    def build_postprocessing_ui(self, parent, palette_colormaps: dict) -> None:
        """Build postprocessing sliders, color controls, and region properties UI.

        Args:
            parent: Parent widget ID to add postprocessing widgets to
            palette_colormaps: Dict mapping palette names to colormap tags
        """
        if dpg is None:
            return

        # Text color hierarchy
        LABEL_TEXT = (150, 150, 150)  # Readable but slightly dimmed

        # Colors section (collapsible)
        with dpg.collapsing_header(label="Colors", default_open=True, parent=parent):
            # Color mode toggle
            with dpg.group(horizontal=True):
                dpg.add_text("Mode:", color=LABEL_TEXT)
                dpg.add_radio_button(
                    items=["Palette", "Expressions"],
                    default_value="Palette",
                    horizontal=True,
                    callback=self.on_color_mode_change,
                    tag="color_mode_radio",
                )

            dpg.add_spacer(height=10)

            # Palette mode container (shown by default)
            with dpg.group(tag="palette_mode_group"):
                # Context header - ALWAYS visible, text changes based on mode
                dpg.add_text("Global Settings", tag="context_header_text", color=(150, 200, 255))

                # Region controls line - ONLY shown when boundary selected
                with dpg.group(horizontal=True, tag="region_controls_line", show=False):
                    dpg.add_radio_button(
                        items=["Surface", "Interior"],
                        default_value="Surface",
                        horizontal=True,
                        callback=self.on_region_toggle,
                        tag="region_toggle_radio",
                    )

                # Placeholder spacer - shown in global mode to reserve same vertical space
                dpg.add_spacer(height=22, tag="region_controls_placeholder", show=True)

                dpg.add_spacer(height=5)

                # Global palette UI (shown in global mode)
                with dpg.group(tag="global_palette_group"):
                    self._build_global_palette_ui("global_palette_group", palette_colormaps)

                # Region palette UI (shown in boundary mode)
                with dpg.group(tag="region_palette_group", show=False):
                    self._build_region_palette_ui("region_palette_group", palette_colormaps)

                # Inline palette editor (hidden until Edit is activated)
                self._build_palette_editor_ui("palette_mode_group")

                # Sliders (fixed position)
                dpg.add_spacer(height=10)

                self.postprocess_clip_low_slider_id = dpg.add_slider_float(
                    label="Clip low %",
                    default_value=self.app.state.display_settings.clip_low_percent,
                    min_value=0.0,
                    max_value=defaults.MAX_CLIP_PERCENT,
                    format="%.2f%%",
                    callback=self.on_clip_low_slider,
                    width=200,
                    tag="clip_low_slider",
                )
                self.postprocess_clip_high_slider_id = dpg.add_slider_float(
                    label="Clip high %",
                    default_value=self.app.state.display_settings.clip_high_percent,
                    min_value=0.0,
                    max_value=defaults.MAX_CLIP_PERCENT,
                    format="%.2f%%",
                    callback=self.on_clip_high_slider,
                    width=200,
                    tag="clip_high_slider",
                )

                self.postprocess_brightness_slider_id = dpg.add_slider_float(
                    label="Brightness",
                    default_value=self.app.state.display_settings.brightness,
                    min_value=defaults.MIN_BRIGHTNESS,
                    max_value=defaults.MAX_BRIGHTNESS,
                    format="%.2f",
                    callback=self.on_brightness_slider,
                    width=200,
                    tag="brightness_slider",
                )
                self.postprocess_contrast_slider_id = dpg.add_slider_float(
                    label="Contrast",
                    default_value=self.app.state.display_settings.contrast,
                    min_value=defaults.MIN_CONTRAST,
                    max_value=defaults.MAX_CONTRAST,
                    format="%.2f",
                    callback=self.on_contrast_slider,
                    width=200,
                    tag="contrast_slider",
                )
                self.postprocess_gamma_slider_id = dpg.add_slider_float(
                    label="Gamma",
                    default_value=self.app.state.display_settings.gamma,
                    min_value=defaults.MIN_GAMMA,
                    max_value=defaults.MAX_GAMMA,
                    format="%.2f",
                    callback=self.on_gamma_slider,
                    width=200,
                    tag="gamma_slider",
                )

                # Lightness expression (palette mode only)
                dpg.add_spacer(height=12)
                dpg.add_separator()
                dpg.add_spacer(height=8)

                with dpg.group(horizontal=True):
                    lightness_expr_lbl = dpg.add_text("Lightness expr", color=(150, 150, 150), tag="lightness_expr_label")
                    with dpg.tooltip(lightness_expr_lbl):
                        dpg.add_text("Custom expression for lightness mapping.\nUse 'mag', 'lic', 'angle', etc.\nExample: clipnorm(mag, 1, 99)")
                    dpg.add_spacer(width=10)
                    # Enable checkbox (for global mode)
                    self.lightness_expr_checkbox_id = dpg.add_checkbox(
                        label="",
                        default_value=self.app.state.display_settings.lightness_expr is not None,
                        callback=self.on_lightness_expr_toggle,
                        tag="lightness_expr_checkbox",
                    )

                # Global/Custom toggle (only visible in region mode)
                with dpg.group(tag="lightness_expr_mode_toggle", show=False):
                    dpg.add_radio_button(
                        items=["Global", "Custom"],
                        default_value="Global",
                        horizontal=True,
                        callback=self.on_lightness_expr_mode_change,
                        tag="lightness_expr_mode_radio",
                    )

                with dpg.group(tag="lightness_expr_group", show=self.app.state.display_settings.lightness_expr is not None):
                    self.lightness_expr_input_id = dpg.add_input_text(
                        default_value=self.app.state.display_settings.lightness_expr or "clipnorm(mag, 1, 99)",
                        width=200,
                        callback=self.on_lightness_expr_change,
                        on_enter=False,
                        tag="lightness_expr_input",
                        hint="e.g. clipnorm(mag, 1, 99)",
                    )

                # Saturation slider (post-colorization chroma multiplier)
                dpg.add_slider_float(
                    label="Saturation",
                    default_value=self.app.state.display_settings.saturation,
                    min_value=0.0,
                    max_value=2.0,
                    format="%.2f",
                    callback=self.on_saturation_change,
                    tag="saturation_slider",
                    width=200,
                    clamped=True,
                )

                self._update_palette_preview_buttons()
                self._refresh_histogram()

                dpg.add_spacer(height=8)

            # Expressions mode container (hidden by default)
            with dpg.group(tag="expressions_mode_group", show=False):
                self._build_expression_editor_ui("expressions_mode_group")

        # Region effects section (smear) - shown when any region is selected
        with dpg.collapsing_header(label="Effects", default_open=True,
                                   tag="effects_header", parent=parent, show=False):
            dpg.add_text("Smear", tag="smear_label")
            self.smear_enabled_checkbox_id = dpg.add_checkbox(
                label="Enable smear",
                callback=self.on_smear_enabled,
                tag="smear_enabled_checkbox",
            )
            self.smear_sigma_slider_id = dpg.add_slider_float(
                label="Blur strength",
                min_value=defaults.MIN_SMEAR_SIGMA,
                max_value=defaults.MAX_SMEAR_SIGMA,
                format="%.4f",
                callback=self.on_smear_sigma,
                tag="smear_sigma_slider",
                width=200,
                clamped=True,
            )

    def _build_global_palette_ui(self, parent, palette_colormaps: dict) -> None:
        """Build global palette selection UI with popup menu.

        Args:
            parent: Parent widget ID
            palette_colormaps: Dict mapping palette names to colormap tags
        """
        if dpg is None:
            return

        palette_names = list(list_color_palettes())

        # Palette label and button showing current selection
        LABEL_TEXT = (150, 150, 150)
        dpg.add_text("Palette", parent=parent, color=LABEL_TEXT)

        # Palette preview button (gradient + label)
        initial_label = self.app.state.display_settings.palette if self.app.state.display_settings.color_enabled else "Grayscale"
        global_palette_button = dpg.add_colormap_button(
            label=initial_label,
            width=self.palette_preview_width,
            height=self.palette_preview_height,
            tag="global_palette_button",
            parent=parent,
        )
        self.global_palette_preview_button_id = global_palette_button

        # Popup menu for global palette selection
        with dpg.popup(global_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="global_palette_popup"):
            dpg.add_text("Palette actions:")
            dpg.add_button(
                label="Edit current",
                width=350,
                height=30,
                callback=self.on_palette_edit,
                user_data=False,
                tag="global_palette_edit_btn",
            )
            dpg.add_button(
                label="Duplicate current",
                width=350,
                height=30,
                callback=self.on_palette_duplicate,
                user_data=False,
                tag="global_palette_duplicate_btn",
            )
            dpg.add_separator()
            dpg.add_text("Select a palette (right-click to delete):")
            dpg.add_separator()

            # Add "Grayscale (No Color)" option at top
            dpg.add_button(
                label="⊙ Grayscale (No Color)",
                width=350,
                height=30,
                callback=self.on_global_grayscale,
                tag="global_grayscale_btn",
            )
            dpg.add_separator()

            with dpg.child_window(width=380, height=300, tag="global_palette_scrolling_window"):
                for palette_name in palette_names:
                    colormap_tag = palette_colormaps[palette_name]

                    # Create a group for each palette entry (colormap button + delete button)
                    with dpg.group(horizontal=True):
                        btn = dpg.add_colormap_button(
                            label=palette_name,
                            width=310,
                            height=25,
                            callback=self.on_global_palette_button,
                            user_data=palette_name,
                            tag=f"global_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                        )
                        dpg.bind_colormap(btn, colormap_tag)

                        # Add delete button that opens separate modal
                        dpg.add_button(
                            label="X",
                            width=30,
                            height=25,
                            callback=lambda s, a, u: self._open_delete_confirmation(u),
                            user_data=palette_name
                        )

        dpg.add_spacer(height=6, parent=parent)
        dpg.add_text("Intensity (post clip/contrast/gamma)", parent=parent, color=LABEL_TEXT)
        self.global_hist_drawlist_id = dpg.add_drawlist(
            width=self.palette_preview_width,
            height=self.palette_hist_height,
            tag="global_hist_drawlist",
            parent=parent,
        )

    def _build_region_palette_ui(self, parent, palette_colormaps: dict) -> None:
        """Build region palette selection UI with popup menu.

        Args:
            parent: Parent widget ID
            palette_colormaps: Dict mapping palette names to colormap tags
        """
        if dpg is None:
            return

        palette_names = list(list_color_palettes())

        # Region palette label and button showing current selection
        LABEL_TEXT = (150, 150, 150)
        dpg.add_text("Region Palette", parent=parent, color=LABEL_TEXT)

        # Palette preview button (gradient + label)
        region_palette_button = dpg.add_colormap_button(
            label="Global",
            width=self.palette_preview_width,
            height=self.palette_preview_height,
            tag="region_palette_button",
            parent=parent,
        )
        self.region_palette_preview_button_id = region_palette_button

        # Popup menu for region palette selection
        with dpg.popup(region_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="region_palette_popup"):
            dpg.add_text("Palette actions:")
            dpg.add_button(
                label="Edit current",
                width=350,
                height=30,
                callback=self.on_palette_edit,
                user_data=True,
                tag="region_palette_edit_btn",
            )
            dpg.add_button(
                label="Duplicate current",
                width=350,
                height=30,
                callback=self.on_palette_duplicate,
                user_data=True,
                tag="region_palette_duplicate_btn",
            )
            dpg.add_separator()
            dpg.add_text("Select palette (also enables slider override):")
            dpg.add_separator()

            # Add "Global" option at top (disables override)
            dpg.add_button(
                label="⊙ Global (no override)",
                width=350,
                height=30,
                callback=self.on_region_use_global,
                tag="region_use_global_btn",
            )
            dpg.add_separator()

            with dpg.child_window(width=380, height=250, tag="region_palette_scrolling_window"):
                for palette_name in palette_names:
                    colormap_tag = palette_colormaps[palette_name]
                    btn = dpg.add_colormap_button(
                        label=palette_name,
                        width=350,
                        height=25,
                        callback=self.on_region_palette_button,
                        user_data=palette_name,
                        tag=f"region_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                    )
                    dpg.bind_colormap(btn, colormap_tag)

        dpg.add_spacer(height=6, parent=parent)
        dpg.add_text("Intensity (post clip/contrast/gamma)", parent=parent, color=LABEL_TEXT)
        self.region_hist_drawlist_id = dpg.add_drawlist(
            width=self.palette_preview_width,
            height=self.palette_hist_height,
            tag="region_hist_drawlist",
            parent=parent,
        )

    def _build_palette_editor_ui(self, parent) -> None:
        """Build the inline palette editor shell (phase 3)."""
        if dpg is None:
            return

        with dpg.group(tag="palette_editor_group", show=False, parent=parent) as group:
            self.palette_editor_group_id = group

            with dpg.group(horizontal=True):
                self.palette_editor_title_id = dpg.add_text(
                    "Editing: (none)",
                    tag="palette_editor_title",
                    color=(150, 200, 255),
                )
                dpg.add_spacer(width=8)
                self.palette_editor_done_button_id = dpg.add_button(
                    label="Done",
                    width=60,
                    callback=self.on_palette_editor_done,
                    tag="palette_editor_done_btn",
                )

            dpg.add_spacer(height=6)
            dpg.add_separator()
            dpg.add_spacer(height=6)

            dpg.add_text("Gradient", color=(150, 150, 150))
            self.palette_editor_gradient_drawlist_id = dpg.add_drawlist(
                width=self.palette_editor_width,
                height=self.palette_editor_gradient_height,
                tag="palette_editor_gradient_drawlist",
            )

            dpg.add_spacer(height=6)
            dpg.add_text("Chroma / Hue (C x H)", color=(150, 150, 150))
            self.palette_editor_slice_drawlist_id = dpg.add_drawlist(
                width=self.palette_editor_width,
                height=self.palette_editor_slice_height,
                tag="palette_editor_slice_drawlist",
            )

            dpg.add_spacer(height=6)
            dpg.add_text("Lightness (L)", color=(150, 150, 150))
            slider_width = max(140, self.palette_editor_width - 60)
            self.palette_editor_lbar_width = slider_width
            self.palette_editor_l_gradient_drawlist_id = dpg.add_drawlist(
                width=self.palette_editor_lbar_width,
                height=self.palette_editor_lbar_height,
                tag="palette_editor_l_gradient_drawlist",
            )

            dpg.add_spacer(height=4)
            self.palette_editor_l_slider_id = dpg.add_slider_float(
                label="L",
                default_value=0.5,
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
                width=slider_width,
                enabled=False,
                callback=self.on_palette_editor_l_slider,
                tag="palette_editor_l_slider",
            )
            self.palette_editor_c_slider_id = dpg.add_slider_float(
                label="C",
                default_value=0.0,
                min_value=0.0,
                max_value=0.4,
                format="%.4f",
                width=slider_width,
                enabled=False,
                callback=self.on_palette_editor_c_slider,
                tag="palette_editor_c_slider",
            )
            self.palette_editor_h_slider_id = dpg.add_slider_float(
                label="H",
                default_value=0.0,
                min_value=0.0,
                max_value=360.0,
                format="%.1f",
                width=slider_width,
                enabled=False,
                callback=self.on_palette_editor_h_slider,
                tag="palette_editor_h_slider",
            )

            dpg.add_spacer(height=6)
            dpg.add_text("Chroma interpolation", color=(150, 150, 150))
            self.palette_editor_interp_mode_id = dpg.add_radio_button(
                items=["Relative", "Absolute"],
                default_value="Relative",
                horizontal=True,
                callback=self.on_palette_editor_interp_mode,
                tag="palette_editor_interp_mode",
            )

            dpg.add_spacer(height=6)

        self._draw_palette_editor_placeholders()

    def _draw_palette_editor_placeholders(self) -> None:
        """Draw placeholder panels for the palette editor."""
        if dpg is None:
            return

        def _draw_panel(drawlist_id: Optional[int], width: int, height: int) -> None:
            if drawlist_id is None or not dpg.does_item_exist(drawlist_id):
                return
            dpg.delete_item(drawlist_id, children_only=True)
            dpg.draw_rectangle(
                (0, 0),
                (width, height),
                color=(70, 70, 70, 255),
                fill=(25, 25, 25, 255),
                thickness=1,
                parent=drawlist_id,
            )

        _draw_panel(self.palette_editor_gradient_drawlist_id, self.palette_editor_width, self.palette_editor_gradient_height)
        _draw_panel(self.palette_editor_slice_drawlist_id, self.palette_editor_width, self.palette_editor_slice_height)
        _draw_panel(self.palette_editor_l_gradient_drawlist_id, self.palette_editor_width, self.palette_editor_lbar_height)

    def _rebuild_palette_editor_grids(self) -> None:
        """Rebuild precomputed grids for the CxH slice."""
        self.palette_editor_c_max_display = 1.0 if self.palette_editor_relative_chroma else self.palette_editor_c_max_absolute
        self.palette_editor_h_grid = np.linspace(
            0.0,
            360.0,
            max(1, int(self.palette_editor_width)),
            endpoint=False,
            dtype=np.float32,
        )
        self.palette_editor_c_grid = np.linspace(
            0.0,
            self.palette_editor_c_max_display,
            max(1, int(self.palette_editor_slice_height)),
            dtype=np.float32,
        )
        self.palette_editor_h_mesh, self.palette_editor_c_mesh = np.meshgrid(
            self.palette_editor_h_grid,
            self.palette_editor_c_grid,
        )

    def _ensure_palette_editor_handlers(self) -> None:
        if dpg is None or self.palette_editor_handler_registry_id is not None:
            return

        with dpg.handler_registry() as handler_reg:
            self.palette_editor_handler_registry_id = handler_reg
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._on_palette_editor_mouse_down,
            )
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._on_palette_editor_mouse_drag,
                threshold=0.0,
            )
            dpg.add_mouse_release_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._on_palette_editor_mouse_release,
            )
            dpg.add_mouse_double_click_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._on_palette_editor_mouse_double_click,
            )

    def _set_palette_editor_controls_enabled(self, enabled: bool) -> None:
        if dpg is None:
            return

        if self.palette_editor_l_slider_id is not None and dpg.does_item_exist(self.palette_editor_l_slider_id):
            dpg.configure_item(self.palette_editor_l_slider_id, enabled=enabled)
        if self.palette_editor_c_slider_id is not None and dpg.does_item_exist(self.palette_editor_c_slider_id):
            dpg.configure_item(self.palette_editor_c_slider_id, enabled=enabled)
        if self.palette_editor_h_slider_id is not None and dpg.does_item_exist(self.palette_editor_h_slider_id):
            dpg.configure_item(self.palette_editor_h_slider_id, enabled=enabled)

    def _palette_editor_item_active(self, item_id: Optional[int]) -> bool:
        if dpg is None or item_id is None:
            return False
        if not dpg.does_item_exist(item_id):
            return False
        return bool(dpg.is_item_active(item_id))

    def _palette_editor_any_slider_active(self) -> bool:
        return (
            self._palette_editor_item_active(self.palette_editor_l_slider_id)
            or self._palette_editor_item_active(self.palette_editor_c_slider_id)
            or self._palette_editor_item_active(self.palette_editor_h_slider_id)
        )

    def _palette_editor_get_stop_index(self, stop_id: int) -> Optional[int]:
        for idx, stop in enumerate(self.palette_editor_stops):
            if stop["id"] == stop_id:
                return idx
        return None

    def _palette_editor_get_selected_stop(self) -> Optional[dict]:
        if self.palette_editor_selected_stop_id is None:
            return None
        idx = self._palette_editor_get_stop_index(self.palette_editor_selected_stop_id)
        if idx is None:
            return None
        return self.palette_editor_stops[idx]

    def _palette_editor_sort_stops(self) -> None:
        self.palette_editor_stops.sort(key=lambda s: s["pos"])

    def _palette_editor_stop_to_x(self, pos: float) -> float:
        bar_left = self.palette_editor_gradient_padding
        bar_right = self.palette_editor_width - self.palette_editor_gradient_padding
        if bar_right <= bar_left:
            return bar_left
        return bar_left + pos * (bar_right - bar_left)

    def _palette_editor_x_to_pos(self, x: float) -> float:
        bar_left = self.palette_editor_gradient_padding
        bar_right = self.palette_editor_width - self.palette_editor_gradient_padding
        if bar_right <= bar_left:
            return 0.0
        return float(np.clip((x - bar_left) / (bar_right - bar_left), 0.0, 1.0))

    def _palette_editor_drawlist_local_coords(
        self,
        drawlist_id: Optional[int],
        mouse_x: float,
        mouse_y: float,
        clamp: bool = False,
    ) -> Optional[tuple[float, float]]:
        if dpg is None or drawlist_id is None or not dpg.does_item_exist(drawlist_id):
            return None

        rect_min = dpg.get_item_rect_min(drawlist_id)
        rect_max = dpg.get_item_rect_max(drawlist_id)

        inside = rect_min[0] <= mouse_x <= rect_max[0] and rect_min[1] <= mouse_y <= rect_max[1]
        if not inside and not clamp:
            return None

        local_x = mouse_x - rect_min[0]
        local_y = mouse_y - rect_min[1]

        if clamp:
            local_x = float(np.clip(local_x, 0.0, rect_max[0] - rect_min[0]))
            local_y = float(np.clip(local_y, 0.0, rect_max[1] - rect_min[1]))

        return float(local_x), float(local_y)

    def _palette_editor_hit_test_handle(self, local_x: float, local_y: float) -> Optional[int]:
        best_idx = None
        best_dist = float(self.palette_editor_handle_hit_radius)

        for idx, stop in enumerate(self.palette_editor_stops):
            x = self._palette_editor_stop_to_x(stop["pos"])
            y = self.palette_editor_handle_center_y
            dist = float(np.hypot(local_x - x, local_y - y))
            if dist <= self.palette_editor_handle_hit_radius and dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx

    def _palette_editor_chroma_rel_to_abs(self, L: float, H: float, C_rel: float) -> float:
        max_c = max_chroma_fast(
            np.array(L, dtype=np.float32),
            np.array(H, dtype=np.float32),
        )
        max_c_val = float(max_c)
        if max_c_val <= 0.0:
            return 0.0
        return float(np.clip(C_rel, 0.0, 1.0) * max_c_val)

    def _palette_editor_chroma_abs_to_rel(self, L: float, H: float, C_abs: float) -> float:
        max_c = max_chroma_fast(
            np.array(L, dtype=np.float32),
            np.array(H, dtype=np.float32),
        )
        max_c_val = float(max_c)
        if max_c_val <= 0.0:
            return 0.0
        return float(np.clip(C_abs / max_c_val, 0.0, 1.0))

    def _palette_editor_chroma_rel_to_abs_array(
        self,
        L_arr: np.ndarray,
        H_arr: np.ndarray,
        C_rel_arr: np.ndarray,
    ) -> np.ndarray:
        max_c = max_chroma_fast(L_arr, H_arr)
        return np.clip(C_rel_arr, 0.0, 1.0) * max_c

    def _palette_editor_chroma_abs_to_rel_array(
        self,
        L_arr: np.ndarray,
        H_arr: np.ndarray,
        C_abs_arr: np.ndarray,
    ) -> np.ndarray:
        max_c = max_chroma_fast(L_arr, H_arr)
        safe = np.where(max_c <= 0.0, 0.0, C_abs_arr / max_c)
        return np.clip(safe, 0.0, 1.0)

    def _palette_editor_oklch_to_rgb(self, L: float, C: float, H: float) -> tuple[float, float, float]:
        if self.palette_editor_relative_chroma:
            C_abs = self._palette_editor_chroma_rel_to_abs(L, H, C)
        else:
            C_abs = float(max(C, 0.0))
        rgb = gamut_map_to_srgb(
            np.array([L], dtype=np.float32),
            np.array([C_abs], dtype=np.float32),
            np.array([H], dtype=np.float32),
            method="compress",
        )
        rgb = np.clip(np.array(rgb, dtype=np.float32), 0.0, 1.0)
        return float(rgb[0, 0]), float(rgb[0, 1]), float(rgb[0, 2])

    def _palette_editor_oklch_to_rgb255(self, L: float, C: float, H: float) -> tuple[int, int, int]:
        r, g, b = self._palette_editor_oklch_to_rgb(L, C, H)
        return int(r * 255), int(g * 255), int(b * 255)

    def _palette_editor_is_in_gamut(self, L: float, C: float, H: float) -> bool:
        if self.palette_editor_relative_chroma:
            return C <= 1.0 + 1e-6
        max_c = max_chroma_fast(
            np.array(L, dtype=np.float32),
            np.array(H, dtype=np.float32),
        )
        return C <= float(max_c) + 1e-6

    def _palette_editor_update_c_slider_range(self) -> None:
        if dpg is None or self.palette_editor_c_slider_id is None:
            return
        if not dpg.does_item_exist(self.palette_editor_c_slider_id):
            return
        max_value = 1.0 if self.palette_editor_relative_chroma else self.palette_editor_c_max_absolute
        fmt = "%.3f" if self.palette_editor_relative_chroma else "%.4f"
        dpg.configure_item(self.palette_editor_c_slider_id, max_value=max_value, format=fmt)

    def _palette_editor_default_stops(self) -> list[dict]:
        c0 = 0.08
        c1 = 0.06
        if self.palette_editor_relative_chroma:
            c0 = self._palette_editor_chroma_abs_to_rel(0.20, 270.0, c0)
            c1 = self._palette_editor_chroma_abs_to_rel(0.92, 60.0, c1)
        return [
            {"id": 0, "pos": 0.0, "L": 0.20, "C": c0, "H": 270.0},
            {"id": 1, "pos": 1.0, "L": 0.92, "C": c1, "H": 60.0},
        ]

    def _palette_editor_convert_rgb_stops(self, stops: list[dict]) -> list[dict]:
        if not stops:
            return []

        positions = []
        colors = []
        missing_pos = False
        for stop in stops:
            if "pos" not in stop:
                missing_pos = True
                break
            positions.append(float(stop["pos"]))
            colors.append([
                float(stop["r"]),
                float(stop["g"]),
                float(stop["b"]),
            ])

        if missing_pos:
            count = len(stops)
            if count == 1:
                positions = [0.0]
            else:
                positions = list(np.linspace(0.0, 1.0, count, dtype=np.float32))
            colors = [
                [float(stop["r"]), float(stop["g"]), float(stop["b"])]
                for stop in stops
            ]

        positions_arr = np.array(positions, dtype=np.float32)
        colors_arr = np.array(colors, dtype=np.float32)
        if positions_arr.size == 0 or colors_arr.size == 0:
            return []

        order = np.argsort(positions_arr)
        positions_arr = positions_arr[order]
        colors_arr = colors_arr[order]
        positions_arr = np.clip(positions_arr, 0.0, 1.0)

        if colors_arr.shape[0] > self.palette_editor_max_stops:
            if positions_arr[0] > 0.0:
                positions_arr = np.insert(positions_arr, 0, 0.0)
                colors_arr = np.vstack([colors_arr[0], colors_arr])
            if positions_arr[-1] < 1.0:
                positions_arr = np.append(positions_arr, 1.0)
                colors_arr = np.vstack([colors_arr, colors_arr[-1]])

            sample_positions = np.linspace(
                0.0,
                1.0,
                self.palette_editor_max_stops,
                dtype=np.float32,
            )
            sampled = np.stack(
                [
                    np.interp(sample_positions, positions_arr, colors_arr[:, 0]),
                    np.interp(sample_positions, positions_arr, colors_arr[:, 1]),
                    np.interp(sample_positions, positions_arr, colors_arr[:, 2]),
                ],
                axis=1,
            )
            positions_arr = sample_positions
            colors_arr = sampled

        rgb = np.clip(colors_arr, 0.0, 1.0)
        L, C, H = srgb_to_oklch(rgb)
        C_rel = self._palette_editor_chroma_abs_to_rel_array(L, H, C)

        converted = []
        for idx, pos in enumerate(positions_arr):
            converted.append({
                "id": idx,
                "pos": float(pos),
                "L": float(L[idx]),
                "C": float(C_rel[idx]),
                "H": float(H[idx]),
            })
        return converted

    def _palette_editor_load_palette(self, palette_name: str) -> None:
        spec = get_palette_spec(palette_name)
        if spec is None:
            return

        space = spec.get("space", "rgb")
        stops = spec.get("stops", [])
        relative_chroma = bool(spec.get("relative_chroma", True))
        interp_mix = float(spec.get("interp_mix", 1.0))
        interp_mix = 1.0 if interp_mix >= 0.5 else 0.0

        self.palette_editor_relative_chroma = relative_chroma
        self.palette_editor_interp_mix = interp_mix

        if space == "oklch":
            editor_stops: list[dict] = []
            for idx, stop in enumerate(stops):
                try:
                    editor_stops.append({
                        "id": idx,
                        "pos": float(stop["pos"]),
                        "L": float(stop["L"]),
                        "C": float(stop["C"]),
                        "H": float(stop["H"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            # Convert RGB palettes to OKLCH for editing (relative chroma).
            self.palette_editor_relative_chroma = True
            self.palette_editor_interp_mix = 1.0
            editor_stops = self._palette_editor_convert_rgb_stops(stops)

        if len(editor_stops) < 2:
            editor_stops = self._palette_editor_default_stops()

        editor_stops.sort(key=lambda s: s["pos"])

        self.palette_editor_stops = editor_stops
        self.palette_editor_next_stop_id = max(stop["id"] for stop in editor_stops) + 1
        self.palette_editor_selected_stop_id = editor_stops[0]["id"]
        self.palette_editor_dragging_stop_id = None
        self.palette_editor_is_dragging = False
        self.palette_editor_slice_drag_active = False
        self.palette_editor_refresh_pending = False
        self.palette_editor_dirty = False
        self.palette_editor_persist_dirty = False
        self.palette_editor_needs_colormap_rebuild = False

        self._rebuild_palette_editor_grids()
        self._palette_editor_update_c_slider_range()
        self._ensure_palette_editor_handlers()
        self._palette_editor_refresh_visuals()
        self._palette_editor_sync_interp_mode()

    def _palette_editor_interp_label(self) -> str:
        return "Relative" if self.palette_editor_interp_mix >= 0.5 else "Absolute"

    def _palette_editor_sync_interp_mode(self) -> None:
        if dpg is None or self.palette_editor_interp_mode_id is None:
            return
        if not dpg.does_item_exist(self.palette_editor_interp_mode_id):
            return
        dpg.set_value(self.palette_editor_interp_mode_id, self._palette_editor_interp_label())

    def _palette_editor_build_spec(self) -> dict:
        stops = sorted(self.palette_editor_stops, key=lambda s: s["pos"])
        spec_stops = [
            {
                "pos": float(stop["pos"]),
                "L": float(stop["L"]),
                "C": float(stop["C"]),
                "H": float(stop["H"]),
            }
            for stop in stops
        ]
        interp_mix = 1.0 if self.palette_editor_interp_mix >= 0.5 else 0.0
        return {
            "space": "oklch",
            "relative_chroma": bool(self.palette_editor_relative_chroma),
            "interp_mix": float(interp_mix),
            "stops": spec_stops,
        }

    def _palette_editor_commit_palette(self, persist: bool = False) -> bool:
        if self.palette_editor_palette_name is None:
            return False

        runtime_dirty = self.palette_editor_dirty
        needs_persist = persist and self.palette_editor_persist_dirty
        if not runtime_dirty and not needs_persist:
            return False

        spec = self._palette_editor_build_spec()
        with self.app.state_lock:
            set_palette_spec(self.palette_editor_palette_name, spec, persist=persist)
            if runtime_dirty:
                self.app.state.invalidate_base_rgb()

        if runtime_dirty:
            updated = self.app.display_pipeline.texture_manager.update_palette_colormap(self.palette_editor_palette_name)
            if not updated:
                self.palette_editor_needs_colormap_rebuild = True

        self.palette_editor_dirty = False
        if persist:
            self.palette_editor_persist_dirty = False
        return runtime_dirty

    def _request_palette_editor_refresh(self, force: bool = False) -> None:
        if not self.palette_editor_active:
            return

        self.palette_editor_dirty = True
        self.palette_editor_persist_dirty = True
        now = time.time()
        if force or (now - self.palette_editor_last_refresh_time) >= self.palette_editor_refresh_throttle:
            self._apply_palette_editor_refresh()
        else:
            self.palette_editor_refresh_pending = True

    def _apply_palette_editor_refresh(self, persist: bool = False) -> None:
        if not self.palette_editor_active:
            self.palette_editor_refresh_pending = False
            return

        had_persist_changes = persist and (self.palette_editor_dirty or self.palette_editor_persist_dirty)
        changed = self._palette_editor_commit_palette(persist=persist)
        if had_persist_changes:
            self.palette_editor_needs_colormap_rebuild = True
        if changed:
            self.app.display_pipeline.refresh_display()
            self.palette_editor_last_refresh_time = time.time()
        self.palette_editor_refresh_pending = False

    def check_palette_editor_debounce(self) -> None:
        if not self.palette_editor_refresh_pending:
            return
        now = time.time()
        if (now - self.palette_editor_last_refresh_time) >= self.palette_editor_refresh_throttle:
            self._apply_palette_editor_refresh()

    def _finalize_palette_editor_colormaps(self) -> None:
        if not self.palette_editor_needs_colormap_rebuild:
            return
        self.app.display_pipeline.texture_manager.rebuild_colormaps()
        self._rebuild_palette_popup()
        self._update_palette_preview_buttons()
        self.palette_editor_needs_colormap_rebuild = False

    def _palette_editor_generate_gradient_texture(self) -> np.ndarray:
        bar_width = max(1, int(self.palette_editor_width - 2 * self.palette_editor_gradient_padding))
        bar_height = max(1, int(self.palette_editor_gradient_bar_bottom - self.palette_editor_gradient_bar_top))

        lut = build_oklch_lut(
            self.palette_editor_stops,
            size=bar_width,
            relative_chroma=self.palette_editor_relative_chroma,
            interp_mix=self.palette_editor_interp_mix,
        )
        rgba = np.ones((bar_height, bar_width, 4), dtype=np.float32)
        rgba[:, :, :3] = lut[np.newaxis, :, :]
        return rgba

    def _palette_editor_generate_slice_texture(self) -> np.ndarray:
        stop = self._palette_editor_get_selected_stop()
        L_val = float(stop["L"]) if stop else 0.5
        L_arr = np.full_like(self.palette_editor_c_mesh, L_val, dtype=np.float32)

        if self.palette_editor_relative_chroma:
            C_abs = self._palette_editor_chroma_rel_to_abs_array(
                L_arr,
                self.palette_editor_h_mesh,
                self.palette_editor_c_mesh,
            )
        else:
            C_abs = np.clip(self.palette_editor_c_mesh, 0.0, None)

        rgb = gamut_map_to_srgb(L_arr, C_abs, self.palette_editor_h_mesh, method="compress")
        rgba = np.ones((self.palette_editor_slice_height, self.palette_editor_width, 4), dtype=np.float32)
        rgba[..., :3] = np.clip(rgb, 0.0, 1.0)
        return rgba

    def _palette_editor_generate_l_gradient_texture(self) -> np.ndarray:
        width = max(1, int(self.palette_editor_lbar_width))
        height = max(1, int(self.palette_editor_lbar_height))

        stop = self._palette_editor_get_selected_stop()
        C_val = float(stop["C"]) if stop else 0.0
        H_val = float(stop["H"]) if stop else 0.0

        L_arr = np.linspace(0.0, 1.0, width, dtype=np.float32)
        H_arr = np.full(width, H_val, dtype=np.float32)

        if self.palette_editor_relative_chroma:
            C_rel = float(np.clip(C_val, 0.0, 1.0))
            max_c = max_chroma_fast(L_arr, H_arr)
            C_abs = max_c * C_rel
        else:
            C_abs = np.full(width, float(np.clip(C_val, 0.0, self.palette_editor_c_max_absolute)), dtype=np.float32)

        rgb = gamut_map_to_srgb(L_arr, C_abs, H_arr, method="compress")
        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = np.clip(rgb, 0.0, 1.0)[np.newaxis, :, :]
        return rgba

    def _palette_editor_update_gradient_texture(self) -> None:
        if dpg is None:
            return
        texture_registry = self.app.display_pipeline.texture_manager.texture_registry_id
        if texture_registry is None:
            return

        rgba = self._palette_editor_generate_gradient_texture()
        data = rgba.ravel()
        width = rgba.shape[1]
        height = rgba.shape[0]
        if self.palette_editor_gradient_texture_id is None or not dpg.does_item_exist(self.palette_editor_gradient_texture_id):
            self.palette_editor_gradient_texture_id = dpg.add_dynamic_texture(
                width,
                height,
                data,
                parent=texture_registry,
            )
        else:
            dpg.set_value(self.palette_editor_gradient_texture_id, data)

    def _palette_editor_update_slice_texture(self) -> None:
        if dpg is None:
            return
        texture_registry = self.app.display_pipeline.texture_manager.texture_registry_id
        if texture_registry is None:
            return

        rgba = self._palette_editor_generate_slice_texture()
        data = rgba.ravel()
        width = rgba.shape[1]
        height = rgba.shape[0]
        if self.palette_editor_slice_texture_id is None or not dpg.does_item_exist(self.palette_editor_slice_texture_id):
            self.palette_editor_slice_texture_id = dpg.add_dynamic_texture(
                width,
                height,
                data,
                parent=texture_registry,
            )
        else:
            dpg.set_value(self.palette_editor_slice_texture_id, data)

    def _palette_editor_update_l_gradient_texture(self) -> None:
        if dpg is None:
            return
        texture_registry = self.app.display_pipeline.texture_manager.texture_registry_id
        if texture_registry is None:
            return

        rgba = self._palette_editor_generate_l_gradient_texture()
        data = rgba.ravel()
        width = rgba.shape[1]
        height = rgba.shape[0]
        if self.palette_editor_l_gradient_texture_id is None or not dpg.does_item_exist(self.palette_editor_l_gradient_texture_id):
            self.palette_editor_l_gradient_texture_id = dpg.add_dynamic_texture(
                width,
                height,
                data,
                parent=texture_registry,
            )
        else:
            dpg.set_value(self.palette_editor_l_gradient_texture_id, data)

    def _palette_editor_update_gradient_drawlist(self) -> None:
        if dpg is None or self.palette_editor_gradient_drawlist_id is None:
            return
        if not dpg.does_item_exist(self.palette_editor_gradient_drawlist_id):
            return

        dpg.delete_item(self.palette_editor_gradient_drawlist_id, children_only=True)

        bar_left = self.palette_editor_gradient_padding
        bar_right = self.palette_editor_width - self.palette_editor_gradient_padding
        bar_top = self.palette_editor_gradient_bar_top
        bar_bottom = self.palette_editor_gradient_bar_bottom

        if self.palette_editor_gradient_texture_id is not None:
            dpg.draw_image(
                self.palette_editor_gradient_texture_id,
                (bar_left, bar_top),
                (bar_right, bar_bottom),
                parent=self.palette_editor_gradient_drawlist_id,
            )

        dpg.draw_rectangle(
            (bar_left, bar_top),
            (bar_right, bar_bottom),
            color=(100, 100, 100, 255),
            thickness=1,
            parent=self.palette_editor_gradient_drawlist_id,
        )

        for is_selected_pass in (False, True):
            for stop in self.palette_editor_stops:
                is_selected = stop["id"] == self.palette_editor_selected_stop_id
                if is_selected != is_selected_pass:
                    continue

                x = self._palette_editor_stop_to_x(stop["pos"])
                y = self.palette_editor_handle_center_y
                r, g, b = self._palette_editor_oklch_to_rgb255(stop["L"], stop["C"], stop["H"])
                if is_selected:
                    dpg.draw_circle(
                        (x, y),
                        self.palette_editor_handle_radius + 3,
                        color=(200, 200, 200, 220),
                        thickness=2,
                        parent=self.palette_editor_gradient_drawlist_id,
                    )
                dpg.draw_circle(
                    (x, y),
                    self.palette_editor_handle_radius,
                    color=(30, 30, 30, 255),
                    fill=(r, g, b, 255),
                    thickness=1,
                    parent=self.palette_editor_gradient_drawlist_id,
                )

    def _palette_editor_update_slice_drawlist(self) -> None:
        if dpg is None or self.palette_editor_slice_drawlist_id is None:
            return
        if not dpg.does_item_exist(self.palette_editor_slice_drawlist_id):
            return

        dpg.delete_item(self.palette_editor_slice_drawlist_id, children_only=True)
        if self.palette_editor_slice_texture_id is not None:
            dpg.draw_image(
                self.palette_editor_slice_texture_id,
                (0, 0),
                (self.palette_editor_width, self.palette_editor_slice_height),
                parent=self.palette_editor_slice_drawlist_id,
            )

        stop = self._palette_editor_get_selected_stop()
        if stop is None:
            return

        x = float(np.clip((stop["H"] / 360.0) * self.palette_editor_width, 0.0, self.palette_editor_width))
        y = float(np.clip((stop["C"] / self.palette_editor_c_max_display) * self.palette_editor_slice_height, 0.0, self.palette_editor_slice_height))
        color = (240, 240, 240, 220)
        dpg.draw_line(
            (0, y),
            (self.palette_editor_width, y),
            color=color,
            thickness=1,
            parent=self.palette_editor_slice_drawlist_id,
        )
        dpg.draw_line(
            (x, 0),
            (x, self.palette_editor_slice_height),
            color=color,
            thickness=1,
            parent=self.palette_editor_slice_drawlist_id,
        )
        dpg.draw_circle(
            (x, y),
            4,
            color=color,
            thickness=1,
            parent=self.palette_editor_slice_drawlist_id,
        )

    def _palette_editor_update_l_gradient_drawlist(self) -> None:
        if dpg is None or self.palette_editor_l_gradient_drawlist_id is None:
            return
        if not dpg.does_item_exist(self.palette_editor_l_gradient_drawlist_id):
            return

        dpg.delete_item(self.palette_editor_l_gradient_drawlist_id, children_only=True)
        if self.palette_editor_l_gradient_texture_id is not None:
            dpg.draw_image(
                self.palette_editor_l_gradient_texture_id,
                (0, 0),
                (self.palette_editor_lbar_width, self.palette_editor_lbar_height),
                parent=self.palette_editor_l_gradient_drawlist_id,
            )

        dpg.draw_rectangle(
            (0, 0),
            (self.palette_editor_lbar_width, self.palette_editor_lbar_height),
            color=(90, 90, 90, 200),
            thickness=1,
            parent=self.palette_editor_l_gradient_drawlist_id,
        )

    def _palette_editor_sync_controls(self, skip_slider_sync: bool = False) -> None:
        if dpg is None:
            return

        stop = self._palette_editor_get_selected_stop()
        if stop is None:
            self._set_palette_editor_controls_enabled(False)
            return

        self._set_palette_editor_controls_enabled(True)
        if self._palette_editor_any_slider_active():
            skip_slider_sync = True

        if not skip_slider_sync:
            self.palette_editor_syncing = True
            try:
                if (
                    self.palette_editor_l_slider_id is not None
                    and dpg.does_item_exist(self.palette_editor_l_slider_id)
                    and not self._palette_editor_item_active(self.palette_editor_l_slider_id)
                ):
                    dpg.set_value(self.palette_editor_l_slider_id, float(stop["L"]))
                if (
                    self.palette_editor_c_slider_id is not None
                    and dpg.does_item_exist(self.palette_editor_c_slider_id)
                    and not self._palette_editor_item_active(self.palette_editor_c_slider_id)
                ):
                    dpg.set_value(self.palette_editor_c_slider_id, float(stop["C"]))
                if (
                    self.palette_editor_h_slider_id is not None
                    and dpg.does_item_exist(self.palette_editor_h_slider_id)
                    and not self._palette_editor_item_active(self.palette_editor_h_slider_id)
                ):
                    dpg.set_value(self.palette_editor_h_slider_id, float(stop["H"]))
            finally:
                self.palette_editor_syncing = False


    def _palette_editor_refresh_visuals(self, skip_slider_sync: bool = False) -> None:
        self._palette_editor_update_gradient_texture()
        self._palette_editor_update_slice_texture()
        self._palette_editor_update_l_gradient_texture()
        self._palette_editor_update_gradient_drawlist()
        self._palette_editor_update_slice_drawlist()
        self._palette_editor_update_l_gradient_drawlist()
        self._palette_editor_sync_controls(skip_slider_sync=skip_slider_sync)

    def _palette_editor_add_stop(self, pos: float) -> None:
        L, C, H = interpolate_oklch(
            self.palette_editor_stops,
            pos,
            relative_chroma=self.palette_editor_relative_chroma,
            interp_mix=self.palette_editor_interp_mix,
        )
        new_id = self.palette_editor_next_stop_id
        self.palette_editor_next_stop_id += 1
        self.palette_editor_stops.append({
            "id": new_id,
            "pos": float(np.clip(pos, 0.0, 1.0)),
            "L": float(L),
            "C": float(C),
            "H": float(H),
        })
        self.palette_editor_selected_stop_id = new_id
        self._palette_editor_sort_stops()

    def _palette_editor_remove_stop(self, stop_id: int) -> None:
        if len(self.palette_editor_stops) <= 2:
            return
        idx = self._palette_editor_get_stop_index(stop_id)
        if idx is None:
            return
        del self.palette_editor_stops[idx]
        if self.palette_editor_stops:
            new_idx = min(idx, len(self.palette_editor_stops) - 1)
            self.palette_editor_selected_stop_id = self.palette_editor_stops[new_idx]["id"]
        else:
            self.palette_editor_selected_stop_id = None

    def _on_palette_editor_mouse_down(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)

        coords = self._palette_editor_drawlist_local_coords(
            self.palette_editor_gradient_drawlist_id,
            mouse_x,
            mouse_y,
            clamp=False,
        )
        if coords is not None:
            local_x, local_y = coords
            handle_idx = self._palette_editor_hit_test_handle(local_x, local_y)
            if handle_idx is not None:
                self.palette_editor_selected_stop_id = self.palette_editor_stops[handle_idx]["id"]
                self.palette_editor_dragging_stop_id = self.palette_editor_selected_stop_id
                self.palette_editor_is_dragging = True
                self._palette_editor_update_gradient_drawlist()
                self._palette_editor_sync_controls()
                return

        coords = self._palette_editor_drawlist_local_coords(
            self.palette_editor_slice_drawlist_id,
            mouse_x,
            mouse_y,
            clamp=False,
        )
        if coords is not None:
            local_x, local_y = coords
            stop = self._palette_editor_get_selected_stop()
            if stop is None:
                return
            self.palette_editor_slice_drag_active = True
            new_h = float(np.clip((local_x / self.palette_editor_width) * 360.0, 0.0, 360.0))
            new_c = float(np.clip((local_y / self.palette_editor_slice_height) * self.palette_editor_c_max_display, 0.0, self.palette_editor_c_max_display))
            stop["H"] = new_h
            stop["C"] = new_c
            self._palette_editor_refresh_visuals()
            self._request_palette_editor_refresh()

    def _on_palette_editor_mouse_drag(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)

        if self.palette_editor_is_dragging and self.palette_editor_dragging_stop_id is not None:
            coords = self._palette_editor_drawlist_local_coords(
                self.palette_editor_gradient_drawlist_id,
                mouse_x,
                mouse_y,
                clamp=True,
            )
            if coords is None:
                return
            local_x, _ = coords
            idx = self._palette_editor_get_stop_index(self.palette_editor_dragging_stop_id)
            if idx is None:
                return
            new_pos = self._palette_editor_x_to_pos(local_x)
            min_pos = 0.0
            max_pos = 1.0
            if idx > 0:
                min_pos = self.palette_editor_stops[idx - 1]["pos"] + 0.005
            if idx < len(self.palette_editor_stops) - 1:
                max_pos = self.palette_editor_stops[idx + 1]["pos"] - 0.005
            self.palette_editor_stops[idx]["pos"] = float(np.clip(new_pos, min_pos, max_pos))
            self._palette_editor_sort_stops()
            self._palette_editor_update_gradient_texture()
            self._palette_editor_update_gradient_drawlist()
            self._palette_editor_sync_controls()
            self._request_palette_editor_refresh()
            return

        if self.palette_editor_slice_drag_active:
            coords = self._palette_editor_drawlist_local_coords(
                self.palette_editor_slice_drawlist_id,
                mouse_x,
                mouse_y,
                clamp=True,
            )
            if coords is None:
                return
            local_x, local_y = coords
            stop = self._palette_editor_get_selected_stop()
            if stop is None:
                return
            stop["H"] = float(np.clip((local_x / self.palette_editor_width) * 360.0, 0.0, 360.0))
            stop["C"] = float(np.clip((local_y / self.palette_editor_slice_height) * self.palette_editor_c_max_display, 0.0, self.palette_editor_c_max_display))
            self._palette_editor_refresh_visuals()
            self._request_palette_editor_refresh()

    def _on_palette_editor_mouse_release(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return

        if self.palette_editor_is_dragging or self.palette_editor_slice_drag_active:
            self.palette_editor_is_dragging = False
            self.palette_editor_dragging_stop_id = None
            self.palette_editor_slice_drag_active = False
            self._palette_editor_sort_stops()
            self._palette_editor_refresh_visuals()
            if self.palette_editor_refresh_pending or self.palette_editor_dirty:
                self._apply_palette_editor_refresh()

    def _on_palette_editor_mouse_double_click(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        coords = self._palette_editor_drawlist_local_coords(
            self.palette_editor_gradient_drawlist_id,
            mouse_x,
            mouse_y,
            clamp=False,
        )
        if coords is None:
            return
        local_x, local_y = coords

        handle_idx = self._palette_editor_hit_test_handle(local_x, local_y)
        if handle_idx is not None:
            stop_id = self.palette_editor_stops[handle_idx]["id"]
            self._palette_editor_remove_stop(stop_id)
            self._palette_editor_refresh_visuals()
            self._request_palette_editor_refresh(force=True)
            return

        if self.palette_editor_gradient_bar_top <= local_y <= self.palette_editor_gradient_bar_bottom + self.palette_editor_handle_radius:
            pos = self._palette_editor_x_to_pos(local_x)
            self._palette_editor_add_stop(pos)
            self._palette_editor_refresh_visuals()
            self._request_palette_editor_refresh(force=True)

    def on_palette_editor_l_slider(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return
        if self.palette_editor_syncing:
            return
        stop = self._palette_editor_get_selected_stop()
        if stop is None:
            return
        stop["L"] = float(np.clip(float(app_data), 0.0, 1.0))
        self._palette_editor_refresh_visuals(skip_slider_sync=True)
        self._request_palette_editor_refresh()

    def on_palette_editor_c_slider(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return
        if self.palette_editor_syncing:
            return
        stop = self._palette_editor_get_selected_stop()
        if stop is None:
            return
        max_value = 1.0 if self.palette_editor_relative_chroma else self.palette_editor_c_max_absolute
        stop["C"] = float(np.clip(float(app_data), 0.0, max_value))
        self._palette_editor_refresh_visuals(skip_slider_sync=True)
        self._request_palette_editor_refresh()

    def on_palette_editor_h_slider(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active:
            return
        if self.palette_editor_syncing:
            return
        stop = self._palette_editor_get_selected_stop()
        if stop is None:
            return
        stop["H"] = float(np.mod(float(app_data), 360.0))
        self._palette_editor_refresh_visuals(skip_slider_sync=True)
        self._request_palette_editor_refresh()

    def on_palette_editor_interp_mode(self, sender=None, app_data=None) -> None:
        if dpg is None or not self.palette_editor_active or self.palette_editor_interp_mode_id is None:
            return
        mode = dpg.get_value(self.palette_editor_interp_mode_id)
        if mode == "Absolute":
            self.palette_editor_interp_mix = 0.0
        else:
            self.palette_editor_interp_mix = 1.0
        self._palette_editor_refresh_visuals()
        self._request_palette_editor_refresh(force=True)

    def _build_expression_editor_ui(self, parent) -> None:
        """Build the expression editor UI for OKLCH color mapping.

        Args:
            parent: Parent widget ID
        """
        if dpg is None:
            return

        from elliptica.colorspace import list_presets, get_preset, AVAILABLE_VARIABLES, AVAILABLE_FUNCTIONS, PDE_SPECIFIC_VARIABLES

        preset_names = list_presets()

        # Brief explanation
        HELP_COLOR = (140, 140, 140)
        dpg.add_text("Map field data to color using math expressions.", color=HELP_COLOR, parent=parent, wrap=280)
        dpg.add_text("Use variables like 'mag' (field magnitude) and 'angle'.", color=HELP_COLOR, parent=parent, wrap=280)
        dpg.add_spacer(height=8, parent=parent)

        # Preset selector
        with dpg.group(horizontal=True, parent=parent):
            dpg.add_text("Preset:")
            self.expr_preset_combo_id = dpg.add_combo(
                items=preset_names,
                default_value=preset_names[0] if preset_names else "",
                width=-1,  # Fill available width
                callback=self.on_expression_preset_change,
                tag="expr_preset_combo",
            )

        dpg.add_spacer(height=10, parent=parent)

        # L expression - add tooltip
        l_label = dpg.add_text("Lightness (L)  [0-1]", parent=parent)
        with dpg.tooltip(l_label):
            dpg.add_text("Controls brightness. 0 = black, 1 = white.")
        self.expr_L_input_id = dpg.add_input_text(
            default_value="clipnorm(lic, 0.5, 99.5)",
            width=-1,  # Fill available width
            height=45,
            multiline=True,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_L_input",
            parent=parent,
        )

        dpg.add_spacer(height=6, parent=parent)

        # C expression - add tooltip
        c_label = dpg.add_text("Chroma (C)  [0-0.4]", parent=parent)
        with dpg.tooltip(c_label):
            dpg.add_text("Color intensity/saturation. 0 = grayscale.")
        self.expr_C_input_id = dpg.add_input_text(
            default_value="0",
            width=-1,  # Fill available width
            height=45,
            multiline=True,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_C_input",
            parent=parent,
        )

        dpg.add_spacer(height=6, parent=parent)

        # H expression - add tooltip
        h_label = dpg.add_text("Hue (H)  [0-360°]", parent=parent)
        with dpg.tooltip(h_label):
            dpg.add_text("Color hue angle. 0=red, 120=green, 240=blue.")
        self.expr_H_input_id = dpg.add_input_text(
            default_value="0",
            width=-1,  # Fill available width
            height=45,
            multiline=True,
            callback=self.on_expression_change,
            on_enter=False,
            tag="expr_H_input",
            parent=parent,
        )

        # Error display
        self.expr_error_text_id = dpg.add_text(
            "",
            color=(255, 100, 100),
            tag="expr_error_text",
            parent=parent,
            wrap=-1,  # Wrap to available width
        )

        dpg.add_spacer(height=4, parent=parent)

        # Reference section (collapsible)
        with dpg.collapsing_header(label="Expression Reference", default_open=False, parent=parent):
            dpg.add_text("Variables:", color=(150, 200, 255))
            for var_name, var_desc in AVAILABLE_VARIABLES:
                dpg.add_text(f"  {var_name}", color=(200, 200, 200))
                dpg.add_text(f"    {var_desc}", color=(150, 150, 150), wrap=260)

            # PDE-specific variables (dynamic based on active PDE)
            dpg.add_spacer(height=4)
            with dpg.group(tag="pde_specific_vars_group"):
                self._update_pde_specific_vars_display()

            dpg.add_spacer(height=8)
            dpg.add_text("Functions:", color=(150, 200, 255))
            for func_sig, func_desc in AVAILABLE_FUNCTIONS:
                dpg.add_text(f"  {func_sig}", color=(200, 200, 200))
                dpg.add_text(f"    {func_desc}", color=(150, 150, 150), wrap=260)

        # Load first preset
        if preset_names:
            self._load_preset(preset_names[0])

    def _update_pde_specific_vars_display(self) -> None:
        """Update the PDE-specific variables display based on active PDE."""
        if dpg is None:
            return

        # Early exit if the group doesn't exist yet (expression editor not built)
        if not dpg.does_item_exist("pde_specific_vars_group"):
            return

        from elliptica.colorspace import PDE_SPECIFIC_VARIABLES

        # Clear existing content
        dpg.delete_item("pde_specific_vars_group", children_only=True)

        # Get active PDE type
        pde_type = self.app.state.project.pde_type if self.app.state.project else "poisson"
        pde_vars = PDE_SPECIFIC_VARIABLES.get(pde_type, [])

        if pde_vars:
            dpg.add_text(f"  ({pde_type}):", color=(150, 180, 150), parent="pde_specific_vars_group")
            for var_name, var_desc in pde_vars:
                dpg.add_text(f"    {var_name}", color=(180, 200, 180), parent="pde_specific_vars_group")
                dpg.add_text(f"      {var_desc}", color=(140, 150, 140), wrap=250, parent="pde_specific_vars_group")

    def _load_preset(self, preset_name: str) -> None:
        """Load a preset into the expression inputs."""
        if dpg is None:
            return

        from elliptica.colorspace import get_preset

        preset = get_preset(preset_name)
        if preset is None:
            return

        dpg.set_value("expr_L_input", preset.L)
        dpg.set_value("expr_C_input", preset.C)
        dpg.set_value("expr_H_input", preset.H)

        # Clear any error
        dpg.set_value("expr_error_text", "")

        # Trigger update
        self._update_color_config_from_expressions()

    def _get_palette_preview_state(self, for_region: bool) -> tuple[str, Optional[str]]:
        """Return label + colormap tag for the palette preview button."""
        texture_manager = self.app.display_pipeline.texture_manager
        grayscale_tag = texture_manager.grayscale_colormap_tag

        with self.app.state_lock:
            if for_region:
                region_style = self._get_current_region_style_unlocked()
                if region_style and region_style.enabled and region_style.use_palette:
                    palette_name = region_style.palette
                    return palette_name, texture_manager.palette_colormaps.get(palette_name) or grayscale_tag

                if self.app.state.display_settings.color_enabled:
                    palette_name = self.app.state.display_settings.palette
                    return palette_name, texture_manager.palette_colormaps.get(palette_name) or grayscale_tag

                return "Grayscale", grayscale_tag

            if self.app.state.display_settings.color_enabled:
                palette_name = self.app.state.display_settings.palette
                return palette_name, texture_manager.palette_colormaps.get(palette_name) or grayscale_tag

            return "Grayscale", grayscale_tag

    def _update_palette_preview_buttons(self) -> None:
        """Refresh palette preview labels and colormap bindings."""
        if dpg is None:
            return

        if self.global_palette_preview_button_id is not None:
            label, tag = self._get_palette_preview_state(for_region=False)
            if dpg.does_item_exist(self.global_palette_preview_button_id):
                dpg.configure_item(self.global_palette_preview_button_id, label=label)
                if tag is not None:
                    dpg.bind_colormap(self.global_palette_preview_button_id, tag)

        if self.region_palette_preview_button_id is not None:
            label, tag = self._get_palette_preview_state(for_region=True)
            if dpg.does_item_exist(self.region_palette_preview_button_id):
                dpg.configure_item(self.region_palette_preview_button_id, label=label)
                if tag is not None:
                    dpg.bind_colormap(self.region_palette_preview_button_id, tag)

        self._update_palette_popup_controls()

    def _get_editable_palette_name(self, for_region: bool) -> Optional[str]:
        """Return palette name if edit/duplicate is valid in this context."""
        with self.app.state_lock:
            if for_region:
                region_style = self._get_current_region_style_unlocked()
                if region_style and region_style.enabled and region_style.use_palette:
                    return region_style.palette
                return None

            if self.app.state.display_settings.color_enabled:
                return self.app.state.display_settings.palette
            return None

    def _update_palette_popup_controls(self) -> None:
        """Enable/disable Edit/Duplicate based on current context."""
        if dpg is None:
            return

        global_editable = self._get_editable_palette_name(for_region=False) is not None
        region_editable = self._get_editable_palette_name(for_region=True) is not None
        if self.palette_editor_active:
            global_editable = False
            region_editable = False

        if dpg.does_item_exist("global_palette_edit_btn"):
            dpg.configure_item("global_palette_edit_btn", enabled=global_editable)
        if dpg.does_item_exist("global_palette_duplicate_btn"):
            dpg.configure_item("global_palette_duplicate_btn", enabled=global_editable)
        if dpg.does_item_exist("region_palette_edit_btn"):
            dpg.configure_item("region_palette_edit_btn", enabled=region_editable)
        if dpg.does_item_exist("region_palette_duplicate_btn"):
            dpg.configure_item("region_palette_duplicate_btn", enabled=region_editable)

    def _set_palette_editor_state(self, active: bool, palette_name: Optional[str] = None, for_region: bool = False) -> None:
        """Show/hide the palette editor and sync controls."""
        if dpg is None:
            return

        self.palette_editor_active = active
        self.palette_editor_palette_name = palette_name if active else None
        self.palette_editor_for_region = for_region if active else False

        if self.palette_editor_group_id is not None and dpg.does_item_exist(self.palette_editor_group_id):
            dpg.configure_item(self.palette_editor_group_id, show=active)

        if self.palette_editor_title_id is not None and dpg.does_item_exist(self.palette_editor_title_id):
            title = f"Editing: {palette_name}" if active and palette_name else "Editing: (none)"
            dpg.set_value(self.palette_editor_title_id, title)

        if self.global_palette_preview_button_id is not None and dpg.does_item_exist(self.global_palette_preview_button_id):
            dpg.configure_item(self.global_palette_preview_button_id, enabled=not active)
        if self.region_palette_preview_button_id is not None and dpg.does_item_exist(self.region_palette_preview_button_id):
            dpg.configure_item(self.region_palette_preview_button_id, enabled=not active)

        if not active:
            self._set_palette_editor_controls_enabled(False)

        self._update_palette_popup_controls()

    def _normalize_unit(self, arr: np.ndarray) -> np.ndarray:
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_max > arr_min:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr, dtype=np.float32)

    def _downsample_histogram_source(self, arr: np.ndarray) -> np.ndarray:
        target_h, target_w = self.hist_target_shape
        if arr.shape[0] == target_h and arr.shape[1] == target_w:
            return arr.astype(np.float32, copy=False)
        try:
            resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            img = Image.fromarray(arr.astype(np.float32, copy=False), mode="F")
            resized = img.resize((target_w, target_h), resample=resample)
            return np.asarray(resized, dtype=np.float32)
        except Exception:
            max_samples = int(self.hist_max_samples)
            if arr.size <= max_samples:
                return arr.astype(np.float32, copy=False)
            step = max(1, int(np.sqrt(arr.size / max_samples)))
            return arr[::step, ::step].astype(np.float32, copy=False)

    def _compute_histogram_values(
        self,
        source: np.ndarray,
        clip_low: float,
        clip_high: float,
        brightness: float,
        contrast: float,
        gamma: float,
        *,
        cached_percentiles: Optional[tuple[float, float]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Compute histogram for post-processed intensity (clip/contrast/gamma)."""
        if source is None:
            return None

        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.float32)
            if mask_arr.shape != source.shape:
                min_h = min(mask_arr.shape[0], source.shape[0])
                min_w = min(mask_arr.shape[1], source.shape[1])
                if min_h <= 0 or min_w <= 0:
                    return None
                source = source[:min_h, :min_w]
                mask_arr = mask_arr[:min_h, :min_w]
        else:
            mask_arr = None

        sample = self._downsample_histogram_source(source)
        weights = None
        if mask_arr is not None:
            mask_sample = self._downsample_histogram_source(mask_arr)
            if mask_sample.shape != sample.shape:
                min_h = min(mask_sample.shape[0], sample.shape[0])
                min_w = min(mask_sample.shape[1], sample.shape[1])
                if min_h <= 0 or min_w <= 0:
                    return None
                sample = sample[:min_h, :min_w]
                mask_sample = mask_sample[:min_h, :min_w]
            weights = np.clip(mask_sample, 0.0, 1.0)
            if float(weights.max()) <= 0.0:
                return np.zeros(self.palette_hist_bins, dtype=np.float32)

        if clip_low > 0.0 or clip_high > 0.0:
            lower = max(0.0, min(clip_low, 100.0))
            upper = max(0.0, min(100.0 - clip_high, 100.0))
            if upper > lower:
                if cached_percentiles is not None:
                    vmin, vmax = cached_percentiles
                else:
                    vmin = float(np.percentile(sample, lower))
                    vmax = float(np.percentile(sample, upper))
            else:
                vmin = float(sample.min())
                vmax = float(sample.max())
        else:
            vmin = float(sample.min())
            vmax = float(sample.max())

        if vmax > vmin:
            norm = (sample - vmin) / (vmax - vmin)
        else:
            norm = self._normalize_unit(sample)
        norm = np.clip(norm, 0.0, 1.0)

        adjusted = norm
        if contrast != 1.0:
            adjusted = (adjusted - 0.5) * contrast + 0.5
            adjusted = np.clip(adjusted, 0.0, 1.0)
        if brightness != 0.0:
            adjusted = np.clip(adjusted + brightness, 0.0, 1.0)
        if gamma != 1.0:
            adjusted = np.clip(adjusted ** gamma, 0.0, 1.0)

        counts, _ = np.histogram(
            adjusted,
            bins=self.palette_hist_bins,
            range=(0.0, 1.0),
            weights=weights,
        )
        counts = counts.astype(np.float32)
        if counts.max() > 0:
            counts /= counts.max()
        if counts.size >= 3:
            kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
            counts = np.convolve(counts, kernel, mode="same")
            if counts.max() > 0:
                counts /= counts.max()
        return counts

    def _update_histogram_drawlist(self, drawlist_id: Optional[int], values: Optional[np.ndarray]) -> None:
        if dpg is None or drawlist_id is None:
            return

        if not dpg.does_item_exist(drawlist_id):
            return

        dpg.delete_item(drawlist_id, children_only=True)

        if values is None:
            return

        width = float(self.palette_preview_width)
        height = float(self.palette_hist_height)
        padding = 6.0
        bar_top = 0.0
        bar_bottom = height - 2.0
        inner_width = max(1.0, width - 2.0 * padding)
        bin_w = inner_width / float(len(values))

        dpg.draw_rectangle(
            (0, bar_top),
            (width, bar_bottom),
            color=(80, 80, 80, 180),
            thickness=1,
            parent=drawlist_id,
        )

        for i, v in enumerate(values):
            x0 = padding + i * bin_w
            x1 = x0 + bin_w
            y1 = bar_bottom
            y0 = bar_bottom - (v * (height - 4.0))
            dpg.draw_rectangle(
                (x0, y0),
                (x1, y1),
                fill=(170, 175, 190, 180),
                color=(0, 0, 0, 0),
                parent=drawlist_id,
            )

    def _refresh_histogram(self) -> None:
        if dpg is None:
            return

        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None or cache.result is None:
                self.global_hist_values = None
                self.region_hist_values = None
                self._update_histogram_drawlist(self.global_hist_drawlist_id, None)
                self._update_histogram_drawlist(self.region_hist_drawlist_id, None)
                self.hist_last_update_time = time.time()
                self.hist_pending_update = False
                return

            source = cache.result.array
            clip_low = float(self.app.state.display_settings.clip_low_percent)
            clip_high = float(self.app.state.display_settings.clip_high_percent)
            brightness = float(self.app.state.display_settings.brightness)
            contrast = float(self.app.state.display_settings.contrast)
            gamma = float(self.app.state.display_settings.gamma)

            cached_percentiles = None
            if cache.lic_percentiles is not None:
                cached_clip = cache.lic_percentiles_clip_range
                if cached_clip is not None:
                    cached_low, cached_high = cached_clip
                    if abs(cached_low - clip_low) < 0.01 and abs(cached_high - clip_high) < 0.01:
                        cached_percentiles = cache.lic_percentiles

            region_mask = None
            region_brightness = brightness
            region_contrast = contrast
            region_gamma = gamma
            boundary_idx = -1
            selected = self.app.state.get_selected()
            boundary_selected = selected is not None and selected.id is not None
            if boundary_selected:
                boundary_idx = self.app.state.get_single_selected_idx()
                region_style = self._get_current_region_style_unlocked()
                if region_style is not None:
                    region_brightness, region_contrast, region_gamma = resolve_region_postprocess_params(
                        region_style,
                        self.app.state.display_settings,
                    )
                if boundary_idx >= 0:
                    if self.app.state.selected_region_type == "surface":
                        masks = cache.boundary_masks
                    else:
                        masks = cache.interior_masks
                    if masks is not None and boundary_idx < len(masks):
                        region_mask = masks[boundary_idx]

        self.global_hist_values = self._compute_histogram_values(
            source,
            clip_low,
            clip_high,
            brightness,
            contrast,
            gamma,
            cached_percentiles=cached_percentiles,
        )

        if boundary_idx >= 0:
            self.region_hist_values = self._compute_histogram_values(
                source,
                clip_low,
                clip_high,
                region_brightness,
                region_contrast,
                region_gamma,
                cached_percentiles=cached_percentiles,
                mask=region_mask,
            )
        else:
            self.region_hist_values = self.global_hist_values

        self._update_histogram_drawlist(self.global_hist_drawlist_id, self.global_hist_values)
        self._update_histogram_drawlist(self.region_hist_drawlist_id, self.region_hist_values)
        self.hist_last_update_time = time.time()
        self.hist_pending_update = False

    def _request_histogram_update(self, force: bool = False) -> None:
        if dpg is None:
            return

        now = time.time()
        if force or (now - self.hist_last_update_time) >= self.hist_debounce_delay:
            self._refresh_histogram()
            return

        self.hist_pending_update = True

    def check_histogram_debounce(self) -> None:
        if not self.hist_pending_update:
            return

        now = time.time()
        if (now - self.hist_last_update_time) >= self.hist_debounce_delay:
            self._refresh_histogram()

    def update_context_ui(self) -> None:
        """Update UI based on current selection context (global vs boundary)."""
        if dpg is None:
            return

        is_boundary_selected = self._is_boundary_selected()

        if is_boundary_selected:
            # Boundary mode
            with self.app.state_lock:
                selected = self.app.state.get_selected()
                boundary_idx = self.app.state.get_single_selected_idx()

            # Update context header text (always visible, just change text)
            region_type = self.app.state.selected_region_type
            region_label = "Surface" if region_type == "surface" else "Interior"
            dpg.set_value("context_header_text", f"Boundary {boundary_idx + 1} - {region_label}")

            # Sync radio button with state
            dpg.set_value("region_toggle_radio", region_label)

            # Show region controls, hide placeholder
            dpg.configure_item("region_controls_line", show=True)
            dpg.configure_item("region_controls_placeholder", show=False)

            # Switch palette UI
            dpg.configure_item("global_palette_group", show=False)
            dpg.configure_item("region_palette_group", show=True)

            # Check if override is enabled (controlled by palette selection)
            has_override = self._has_override_enabled()

            region_style = self._get_current_region_style()

            # Update lightness expression controls for region mode
            # Show Global/Custom toggle (independent of palette override)
            dpg.configure_item("lightness_expr_mode_toggle", show=True)
            dpg.configure_item("lightness_expr_checkbox", show=False)

            has_custom_expr = region_style is not None and region_style.lightness_expr is not None
            dpg.set_value("lightness_expr_mode_radio", "Custom" if has_custom_expr else "Global")

            # Populate cache from existing settings (e.g. loaded from project file)
            if has_custom_expr and selected is not None and selected.id is not None:
                cache_key = (selected.id, self.app.state.selected_region_type)
                if cache_key not in self._cached_custom_lightness_exprs:
                    self._cached_custom_lightness_exprs[cache_key] = region_style.lightness_expr

            # Show input if global expr is enabled OR region has custom expr
            global_expr_enabled = self.app.state.display_settings.lightness_expr is not None
            show_input = global_expr_enabled or has_custom_expr
            dpg.configure_item("lightness_expr_group", show=show_input)

            if has_custom_expr:
                dpg.set_value("lightness_expr_input", region_style.lightness_expr)
                self._set_input_grayed("lightness_expr_input", False)
            else:
                # Show global expr (grayed out)
                global_expr = self.app.state.display_settings.lightness_expr or "clipnorm(mag, 1, 99)"
                dpg.set_value("lightness_expr_input", global_expr)
                self._set_input_grayed("lightness_expr_input", True)

            # Clip sliders always show global, grayed in boundary mode
            self._set_slider_grayed("clip_low_slider", True)
            self._set_slider_grayed("clip_high_slider", True)
            dpg.set_value("clip_low_slider", self.app.state.display_settings.clip_low_percent)
            dpg.set_value("clip_high_slider", self.app.state.display_settings.clip_high_percent)

            # Configure B/C/G sliders - grayed until override enabled
            self._set_slider_grayed("brightness_slider", not has_override)
            self._set_slider_grayed("contrast_slider", not has_override)
            self._set_slider_grayed("gamma_slider", not has_override)

            if has_override and region_style:
                # Show per-region values
                b = region_style.brightness if region_style.brightness is not None else self.app.state.display_settings.brightness
                c = region_style.contrast if region_style.contrast is not None else self.app.state.display_settings.contrast
                g = region_style.gamma if region_style.gamma is not None else self.app.state.display_settings.gamma
            else:
                # Show global values (what's being applied)
                b = self.app.state.display_settings.brightness
                c = self.app.state.display_settings.contrast
                g = self.app.state.display_settings.gamma

            dpg.set_value("brightness_slider", b)
            dpg.set_value("contrast_slider", c)
            dpg.set_value("gamma_slider", g)

            # Show effects section and update smear controls from RegionStyle
            dpg.configure_item("effects_header", show=True)
            if region_style:
                dpg.set_value("smear_enabled_checkbox", region_style.smear_enabled)
                dpg.set_value("smear_sigma_slider", region_style.smear_sigma)
                dpg.configure_item("smear_sigma_slider", show=region_style.smear_enabled)
            else:
                dpg.set_value("smear_enabled_checkbox", False)
                dpg.configure_item("smear_sigma_slider", show=False)

        else:
            # Global mode (no boundary selected)
            dpg.set_value("context_header_text", "Global Settings")

            # Hide region controls, show placeholder (maintains layout)
            dpg.configure_item("region_controls_line", show=False)
            dpg.configure_item("region_controls_placeholder", show=True)

            # Switch palette UI
            dpg.configure_item("global_palette_group", show=True)
            dpg.configure_item("region_palette_group", show=False)

            # Clip sliders normal in global mode
            self._set_slider_grayed("clip_low_slider", False)
            self._set_slider_grayed("clip_high_slider", False)
            dpg.set_value("clip_low_slider", self.app.state.display_settings.clip_low_percent)
            dpg.set_value("clip_high_slider", self.app.state.display_settings.clip_high_percent)

            # B/C/G sliders normal, showing global values
            self._set_slider_grayed("brightness_slider", False)
            self._set_slider_grayed("contrast_slider", False)
            self._set_slider_grayed("gamma_slider", False)
            dpg.set_value("brightness_slider", self.app.state.display_settings.brightness)
            dpg.set_value("contrast_slider", self.app.state.display_settings.contrast)
            dpg.set_value("gamma_slider", self.app.state.display_settings.gamma)

            # Lightness expression - global mode (show Enable checkbox, hide toggle)
            dpg.configure_item("lightness_expr_mode_toggle", show=False)
            dpg.configure_item("lightness_expr_checkbox", show=True)
            dpg.configure_item("lightness_expr_checkbox", enabled=True)
            global_expr_enabled = self.app.state.display_settings.lightness_expr is not None
            dpg.set_value("lightness_expr_checkbox", global_expr_enabled)
            dpg.configure_item("lightness_expr_group", show=global_expr_enabled)
            if global_expr_enabled:
                dpg.set_value("lightness_expr_input", self.app.state.display_settings.lightness_expr)
                # Populate cache from existing state (e.g. loaded from project)
                if self._cached_global_lightness_expr is None:
                    self._cached_global_lightness_expr = self.app.state.display_settings.lightness_expr
            self._set_input_grayed("lightness_expr_input", False)

            # Hide effects section
            dpg.configure_item("effects_header", show=False)

        self._update_palette_preview_buttons()
        self._request_histogram_update()

    def update_region_properties_panel(self) -> None:
        """Update region properties panel based on current selection.

        This is called when selection changes. Delegates to update_context_ui.
        """
        # Apply pending lightness expression updates before clearing them
        # (don't lose user changes when context changes)
        if self.lightness_expr_pending_update:
            if isinstance(self.lightness_expr_pending_target, tuple):
                boundary_id, region_type = self.lightness_expr_pending_target
                self._apply_region_lightness_expr_update(boundary_id, region_type)
            elif self.lightness_expr_pending_target == "global":
                self._apply_lightness_expr_update()

        # Now safe to clear pending states
        self.lightness_expr_pending_update = False
        self.lightness_expr_pending_target = None
        self.smear_pending_value = None
        self.clip_pending_range = None
        self.expr_pending_update = False  # Clear expression editor debounce too
        self.update_context_ui()

    # ------------------------------------------------------------------
    # Region toggle callback
    # ------------------------------------------------------------------

    def on_region_toggle(self, sender=None, app_data=None) -> None:
        """Handle Surface/Interior region toggle."""
        if dpg is None:
            return

        with self.app.state_lock:
            self.app.state.selected_region_type = "surface" if app_data == "Surface" else "interior"
        self.app.canvas_renderer.invalidate_selection_contour()
        self.app.canvas_renderer.mark_dirty()
        self.update_context_ui()

    # ------------------------------------------------------------------
    # Postprocessing slider callbacks
    # ------------------------------------------------------------------

    def on_clip_low_slider(self, sender=None, app_data=None) -> None:
        """Handle low clip slider change with debouncing (percentile computation is expensive at high res)."""
        if dpg is None:
            return

        value = float(app_data)

        # IMMEDIATELY update state so other refreshes use the correct value
        with self.app.state_lock:
            self.app.state.display_settings.clip_low_percent = value
            clip_low = self.app.state.display_settings.clip_low_percent
            clip_high = self.app.state.display_settings.clip_high_percent
            self.app.state.invalidate_base_rgb()

        # Mark pending to trigger refresh after debounce delay
        self.clip_pending_range = (clip_low, clip_high)
        self.clip_last_update_time = time.time()

    def on_clip_high_slider(self, sender=None, app_data=None) -> None:
        """Handle high clip slider change with debouncing (percentile computation is expensive at high res)."""
        if dpg is None:
            return

        value = float(app_data)

        # IMMEDIATELY update state so other refreshes use the correct value
        with self.app.state_lock:
            self.app.state.display_settings.clip_high_percent = value
            clip_low = self.app.state.display_settings.clip_low_percent
            clip_high = self.app.state.display_settings.clip_high_percent
            self.app.state.invalidate_base_rgb()

        # Mark pending to trigger refresh after debounce delay
        self.clip_pending_range = (clip_low, clip_high)
        self.clip_last_update_time = time.time()

    def on_brightness_slider(self, sender=None, app_data=None) -> None:
        """Handle brightness slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        has_override, boundary_id, region_type = self._get_slider_context()

        with self.app.state_lock:
            if has_override and boundary_id is not None:
                actions.set_region_brightness(self.app.state, boundary_id, region_type, value)
            else:
                self.app.state.display_settings.brightness = value
                self.app.state.invalidate_base_rgb()

        self.app.display_pipeline.refresh_display()
        self._request_histogram_update()

    def on_contrast_slider(self, sender=None, app_data=None) -> None:
        """Handle contrast slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        has_override, boundary_id, region_type = self._get_slider_context()

        with self.app.state_lock:
            if has_override and boundary_id is not None:
                actions.set_region_contrast(self.app.state, boundary_id, region_type, value)
            else:
                self.app.state.display_settings.contrast = value
                self.app.state.invalidate_base_rgb()

        self.app.display_pipeline.refresh_display()
        self._request_histogram_update()

    def on_gamma_slider(self, sender=None, app_data=None) -> None:
        """Handle gamma slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        has_override, boundary_id, region_type = self._get_slider_context()

        with self.app.state_lock:
            if has_override and boundary_id is not None:
                actions.set_region_gamma(self.app.state, boundary_id, region_type, value)
            else:
                self.app.state.display_settings.gamma = value
                self.app.state.invalidate_base_rgb()

        self.app.display_pipeline.refresh_display()
        self._request_histogram_update()

    def on_saturation_change(self, sender=None, app_data=None) -> None:
        """Handle saturation slider change (post-colorization chroma multiplier)."""
        if dpg is None:
            return

        value = float(app_data)
        with self.app.state_lock:
            self.app.state.display_settings.saturation = value

        self.app.display_pipeline.refresh_display()

    # ------------------------------------------------------------------
    # Lightness expression callbacks (palette mode)
    # ------------------------------------------------------------------

    def on_lightness_expr_toggle(self, sender=None, app_data=None) -> None:
        """Handle lightness expression checkbox toggle."""
        if dpg is None:
            return

        is_enabled = bool(app_data)

        # Show/hide the input field
        dpg.configure_item("lightness_expr_group", show=is_enabled)

        if is_enabled:
            # Enable - use cached global expr or default (NOT the input field, which may show region expr)
            expr = self._cached_global_lightness_expr or "clipnorm(mag, 1, 99)"
            with self.app.state_lock:
                self.app.state.display_settings.lightness_expr = expr
            # Update input to show the global expression
            dpg.set_value("lightness_expr_input", expr)
        else:
            # Disable - cache current global expr first
            with self.app.state_lock:
                if self.app.state.display_settings.lightness_expr:
                    self._cached_global_lightness_expr = self.app.state.display_settings.lightness_expr
                self.app.state.display_settings.lightness_expr = None

        self.app.display_pipeline.refresh_display()

    def on_lightness_expr_change(self, sender=None, app_data=None) -> None:
        """Handle lightness expression text change (debounced)."""
        if dpg is None:
            return

        # Capture the target NOW based on UI state (radio button), not region_style state
        # The radio button reflects the user's intent directly
        if self._is_boundary_selected():
            mode = dpg.get_value("lightness_expr_mode_radio")
            if mode == "Custom":
                with self.app.state_lock:
                    selected = self.app.state.get_selected()
                    region_type = self.app.state.selected_region_type
                    if selected and selected.id is not None:
                        self.lightness_expr_pending_target = (selected.id, region_type)
                    else:
                        self.lightness_expr_pending_target = "global"
            else:
                self.lightness_expr_pending_target = "global"
        else:
            self.lightness_expr_pending_target = "global"

        # Mark pending update and record time
        self.lightness_expr_pending_update = True
        self.lightness_expr_last_update_time = time.time()

    def check_lightness_expr_debounce(self) -> None:
        """Check if lightness expression update should be applied (called every frame)."""
        if not self.lightness_expr_pending_update:
            return

        current_time = time.time()
        if current_time - self.lightness_expr_last_update_time >= self.expr_debounce_delay:
            # Use the target captured at input time (not current state)
            target = self.lightness_expr_pending_target
            if isinstance(target, tuple):
                # Per-region update
                boundary_id, region_type = target
                self._apply_region_lightness_expr_update(boundary_id, region_type)
            else:
                # Global update
                self._apply_lightness_expr_update()
            self.lightness_expr_pending_update = False
            self.lightness_expr_pending_target = None

    def _apply_lightness_expr_update(self) -> None:
        """Apply the current lightness expression from the input field."""
        if dpg is None:
            return

        expr = dpg.get_value("lightness_expr_input").strip()
        if not expr:
            return

        with self.app.state_lock:
            self.app.state.display_settings.lightness_expr = expr
        # Also update cache so toggling off/on preserves the latest edit
        self._cached_global_lightness_expr = expr

        self.app.display_pipeline.refresh_display()

    # ------------------------------------------------------------------
    # Lightness expression mode (Global/Custom) callback
    # ------------------------------------------------------------------

    def on_lightness_expr_mode_change(self, sender=None, app_data=None) -> None:
        """Handle Global/Custom radio toggle for per-region lightness expression."""
        if dpg is None:
            return

        # Cancel any pending debounced updates - mode is changing
        self.lightness_expr_pending_update = False
        self.lightness_expr_pending_target = None

        is_custom = (app_data == "Custom")

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected is None or selected.id is None:
                return

            # Ensure BoundaryColorSettings exists for this boundary
            from elliptica.app.core import BoundaryColorSettings
            if selected.id not in self.app.state.boundary_color_settings:
                self.app.state.boundary_color_settings[selected.id] = BoundaryColorSettings()

            settings = self.app.state.boundary_color_settings[selected.id]
            region_type = self.app.state.selected_region_type
            region_style = settings.surface if region_type == "surface" else settings.interior
            cache_key = (selected.id, region_type)

            if is_custom:
                # Switch to custom - check cache first, then global, then default
                cached_expr = self._cached_custom_lightness_exprs.get(cache_key)
                if cached_expr is not None:
                    region_style.lightness_expr = cached_expr
                else:
                    global_expr = self.app.state.display_settings.lightness_expr
                    region_style.lightness_expr = global_expr or "clipnorm(mag, 1, 99)"
            else:
                # Switch to global - cache current expr before clearing
                if region_style.lightness_expr is not None:
                    self._cached_custom_lightness_exprs[cache_key] = region_style.lightness_expr
                region_style.lightness_expr = None

        self.update_context_ui()
        self.app.display_pipeline.refresh_display()

    def _apply_region_lightness_expr_update(self, boundary_id: int, region_type: str) -> None:
        """Apply the current per-region lightness expression.

        Args:
            boundary_id: The boundary ID to update
            region_type: "surface" or "interior"
        """
        if dpg is None:
            return

        expr = dpg.get_value("lightness_expr_input").strip()
        if not expr:
            return

        with self.app.state_lock:
            # Ensure settings exist
            from elliptica.app.core import BoundaryColorSettings
            if boundary_id not in self.app.state.boundary_color_settings:
                self.app.state.boundary_color_settings[boundary_id] = BoundaryColorSettings()

            settings = self.app.state.boundary_color_settings[boundary_id]
            region_style = settings.surface if region_type == "surface" else settings.interior

            # Set the expression (we're targeting this region, so apply it)
            region_style.lightness_expr = expr
            # Also update cache so switching Global->Custom restores latest edit
            cache_key = (boundary_id, region_type)
            self._cached_custom_lightness_exprs[cache_key] = expr

        self.app.display_pipeline.refresh_display()

    # ------------------------------------------------------------------
    # Color and palette callbacks
    # ------------------------------------------------------------------

    def on_palette_edit(self, sender=None, app_data=None, user_data=None) -> None:
        """Open the inline palette editor for the active palette."""
        if dpg is None:
            return

        for_region = bool(user_data)
        palette_name = self._get_editable_palette_name(for_region)
        if palette_name is None:
            return

        self._set_palette_editor_state(True, palette_name, for_region)
        self._palette_editor_load_palette(palette_name)
        popup_tag = "region_palette_popup" if for_region else "global_palette_popup"
        if dpg.does_item_exist(popup_tag):
            dpg.configure_item(popup_tag, show=False)

    def on_palette_editor_done(self, sender=None, app_data=None) -> None:
        """Exit palette edit mode."""
        if dpg is None:
            return
        if self.palette_editor_dirty or self.palette_editor_persist_dirty:
            self._apply_palette_editor_refresh(persist=True)
        self._finalize_palette_editor_colormaps()
        self._set_palette_editor_state(False)
        self.palette_editor_refresh_pending = False

    def _generate_duplicate_palette_name(self, base: str) -> str:
        existing = set(list_color_palettes())
        candidate = f"{base} Copy"
        if candidate not in existing:
            return candidate
        index = 2
        while True:
            candidate = f"{base} Copy {index}"
            if candidate not in existing:
                return candidate
            index += 1

    def on_palette_duplicate(self, sender=None, app_data=None, user_data=None) -> None:
        """Duplicate the active palette and enter edit mode on the copy."""
        if dpg is None:
            return

        for_region = bool(user_data)
        if self.palette_editor_active:
            return

        palette_name = self._get_editable_palette_name(for_region)
        if palette_name is None:
            return

        spec = get_palette_spec(palette_name)
        if spec is None:
            return

        new_name = self._generate_duplicate_palette_name(palette_name)
        set_palette_spec(new_name, spec)

        self.app.display_pipeline.texture_manager.rebuild_colormaps()
        self._rebuild_palette_popup()

        with self.app.state_lock:
            if for_region:
                selected = self.app.state.get_selected()
                region_type = self.app.state.selected_region_type
                if selected and selected.id is not None:
                    actions.set_region_palette(self.app.state, selected.id, region_type, new_name)
                    global_b = self.app.state.display_settings.brightness
                    global_c = self.app.state.display_settings.contrast
                    global_g = self.app.state.display_settings.gamma
                    actions.set_region_brightness(self.app.state, selected.id, region_type, global_b)
                    actions.set_region_contrast(self.app.state, selected.id, region_type, global_c)
                    actions.set_region_gamma(self.app.state, selected.id, region_type, global_g)
            else:
                actions.set_color_enabled(self.app.state, True)
                actions.set_palette(self.app.state, new_name)

        self.update_context_ui()
        self._update_palette_preview_buttons()

        popup_tag = "region_palette_popup" if for_region else "global_palette_popup"
        if dpg.does_item_exist(popup_tag):
            dpg.configure_item(popup_tag, show=False)

        self._set_palette_editor_state(True, new_name, for_region)
        self._palette_editor_load_palette(new_name)
        self.app.display_pipeline.refresh_display()

    def on_global_grayscale(self, sender=None, app_data=None) -> None:
        """Handle global 'Grayscale (No Color)' button."""
        if dpg is None:
            return
        if self.palette_editor_active:
            return

        with self.app.state_lock:
            actions.set_color_enabled(self.app.state, False)

        self._update_palette_preview_buttons()
        dpg.configure_item("global_palette_popup", show=False)

        self.app.display_pipeline.refresh_display()

    def on_global_palette_button(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle global colormap button click."""
        if dpg is None or user_data is None:
            return
        if self.palette_editor_active:
            return

        palette_name = user_data
        with self.app.state_lock:
            # Auto-enable color when a palette is selected
            actions.set_color_enabled(self.app.state, True)
            actions.set_palette(self.app.state, palette_name)

        self._update_palette_preview_buttons()
        dpg.configure_item("global_palette_popup", show=False)

        self.app.display_pipeline.refresh_display()

    def on_region_use_global(self, sender=None, app_data=None) -> None:
        """Handle region 'Use Global' button - disables override."""
        if dpg is None:
            return
        if self.palette_editor_active:
            return

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            region_type = self.app.state.selected_region_type
            if selected and selected.id is not None:
                # Disable palette override
                actions.set_region_style_enabled(self.app.state, selected.id, region_type, False)
                # Also clear B/C/G (disables slider override)
                actions.set_region_brightness(self.app.state, selected.id, region_type, None)
                actions.set_region_contrast(self.app.state, selected.id, region_type, None)
                actions.set_region_gamma(self.app.state, selected.id, region_type, None)

        dpg.configure_item("region_palette_popup", show=False)

        self.update_context_ui()  # Update slider states
        self.app.display_pipeline.refresh_display()

    def on_region_palette_button(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle region colormap button click - also enables override."""
        if dpg is None or user_data is None:
            return
        if self.palette_editor_active:
            return

        palette_name = user_data
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            region_type = self.app.state.selected_region_type
            if selected and selected.id is not None:
                # Set palette (this also sets enabled=True)
                actions.set_region_palette(self.app.state, selected.id, region_type, palette_name)
                # Also initialize B/C/G with global values (enables slider override)
                global_b = self.app.state.display_settings.brightness
                global_c = self.app.state.display_settings.contrast
                global_g = self.app.state.display_settings.gamma
                actions.set_region_brightness(self.app.state, selected.id, region_type, global_b)
                actions.set_region_contrast(self.app.state, selected.id, region_type, global_c)
                actions.set_region_gamma(self.app.state, selected.id, region_type, global_g)

        dpg.configure_item("region_palette_popup", show=False)

        self.update_context_ui()  # Update slider states
        self.app.display_pipeline.refresh_display()

    def _ensure_delete_confirmation_modal(self) -> None:
        """Create the delete confirmation modal if it doesn't exist."""
        if dpg is None or self.delete_confirmation_modal_id is not None:
            return

        with dpg.window(
            label="Delete Palette?",
            modal=True,
            show=False,
            tag="delete_palette_modal",
            no_resize=True,
            no_move=False,
            no_close=True,
            no_collapse=True,
            width=350,
            height=150,
            no_open_over_existing_popup=False,
        ) as modal:
            self.delete_confirmation_modal_id = modal

            dpg.add_text("", tag="delete_palette_modal_text")
            dpg.add_text("This cannot be undone.", color=(200, 200, 0))
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Delete",
                    width=75,
                    callback=self._confirm_delete_palette
                )
                dpg.add_button(
                    label="Cancel",
                    width=75,
                    callback=self._cancel_delete_palette
                )

    def _open_delete_confirmation(self, palette_name: str) -> None:
        """Open the delete confirmation modal for a specific palette."""
        if dpg is None:
            return

        self._ensure_delete_confirmation_modal()
        self.pending_delete_palette = palette_name

        dpg.set_value("delete_palette_modal_text", f"Delete '{palette_name}'?")

        # Center the modal on screen
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        modal_width = 350
        modal_height = 150
        pos_x = (viewport_width - modal_width) // 2
        pos_y = (viewport_height - modal_height) // 2

        dpg.configure_item(self.delete_confirmation_modal_id, pos=[pos_x, pos_y])
        dpg.configure_item(self.delete_confirmation_modal_id, show=True)

    def _cancel_delete_palette(self, sender=None, app_data=None) -> None:
        """Cancel palette deletion."""
        if dpg is None or self.delete_confirmation_modal_id is None:
            return
        dpg.configure_item(self.delete_confirmation_modal_id, show=False)
        self.pending_delete_palette = None

    def _confirm_delete_palette(self, sender=None, app_data=None, user_data=None) -> None:
        """Actually delete the palette after confirmation."""
        if dpg is None or self.pending_delete_palette is None:
            return

        palette_name = self.pending_delete_palette
        delete_palette(palette_name)

        # Rebuild DPG colormaps to reflect deletion
        self.app.display_pipeline.texture_manager.rebuild_colormaps()

        # Rebuild the palette popup menu
        self._rebuild_palette_popup()
        self._update_palette_preview_buttons()

        # Close the confirmation modal
        dpg.configure_item(self.delete_confirmation_modal_id, show=False)
        self.pending_delete_palette = None

    def _rebuild_palette_popup(self) -> None:
        """Rebuild palette popup after deletion."""
        if dpg is None:
            return

        # Delete old scrolling window content
        if dpg.does_item_exist("global_palette_scrolling_window"):
            dpg.delete_item("global_palette_scrolling_window", children_only=True)

        # Rebuild colormap buttons
        palette_names = list(list_color_palettes())

        for palette_name in palette_names:
            colormap_tag = self.app.display_pipeline.texture_manager.palette_colormaps.get(palette_name)
            if not colormap_tag:
                continue

            # Create a group for each palette entry (colormap button + delete button)
            with dpg.group(horizontal=True, parent="global_palette_scrolling_window"):
                btn = dpg.add_colormap_button(
                    label=palette_name,
                    width=310,
                    height=25,
                    callback=self.on_global_palette_button,
                    user_data=palette_name,
                    tag=f"global_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                )
                dpg.bind_colormap(btn, colormap_tag)

                # Add delete button that opens separate modal
                dpg.add_button(
                    label="X",
                    width=30,
                    height=25,
                    callback=lambda s, a, u: self._open_delete_confirmation(u),
                    user_data=palette_name
                )

        # Rebuild region palette list too
        if dpg.does_item_exist("region_palette_scrolling_window"):
            dpg.delete_item("region_palette_scrolling_window", children_only=True)

            for palette_name in palette_names:
                colormap_tag = self.app.display_pipeline.texture_manager.palette_colormaps.get(palette_name)
                if not colormap_tag:
                    continue
                btn = dpg.add_colormap_button(
                    label=palette_name,
                    width=350,
                    height=25,
                    callback=self.on_region_palette_button,
                    user_data=palette_name,
                    tag=f"region_palette_btn_{palette_name.replace(' ', '_').replace('&', 'and')}",
                    parent="region_palette_scrolling_window",
                )
                dpg.bind_colormap(btn, colormap_tag)

    # ------------------------------------------------------------------
    # Smear callbacks
    # ------------------------------------------------------------------

    def on_smear_enabled(self, sender=None, app_data=None) -> None:
        """Toggle smear for selected region."""
        if dpg is None:
            return

        with self.app.state_lock:
            region_style = self._get_current_region_style_unlocked()
            if region_style is not None:
                region_style.smear_enabled = bool(app_data)

        self.update_context_ui()
        self.app.display_pipeline.refresh_display()

    def on_smear_sigma(self, sender=None, app_data=None) -> None:
        """Adjust smear blur sigma for selected region (debounced for performance)."""
        if dpg is None:
            return

        # Always update the value immediately (for UI responsiveness)
        with self.app.state_lock:
            region_style = self._get_current_region_style_unlocked()
            if region_style is not None:
                region_style.smear_sigma = float(app_data)

        # Mark that we have a pending update (defers expensive render refresh)
        self.smear_pending_value = float(app_data)

        # Record the time of THIS slider change (not the last render)
        self.smear_last_update_time = time.time()

    def _apply_clip_update(self) -> None:
        """Apply clip refresh (state already updated in slider callbacks)."""
        self.app.display_pipeline.refresh_display()
        self._request_histogram_update(force=True)

    def _apply_smear_update(self) -> None:
        """Apply smear refresh (state already updated in on_smear_sigma)."""
        self.app.display_pipeline.refresh_display()

    def check_clip_debounce(self) -> None:
        """Check if clip update should be applied (called every frame)."""
        if self.clip_pending_range is None:
            return

        current_time = time.time()
        # Only apply if enough time has passed since the last slider movement
        if current_time - self.clip_last_update_time >= self.clip_debounce_delay:
            self._apply_clip_update()
            self.clip_last_update_time = current_time
            self.clip_pending_range = None

    def check_smear_debounce(self) -> None:
        """Check if smear update should be applied (called every frame)."""
        if self.smear_pending_value is None:
            return

        current_time = time.time()
        # Only apply if enough time has passed since the last slider movement
        if current_time - self.smear_last_update_time >= self.smear_debounce_delay:
            self._apply_smear_update()
            self.smear_last_update_time = current_time
            self.smear_pending_value = None

    # ------------------------------------------------------------------
    # Expression editor callbacks
    # ------------------------------------------------------------------

    def on_color_mode_change(self, sender=None, app_data=None) -> None:
        """Handle color mode toggle (Palette / Expressions)."""
        if dpg is None:
            return

        mode = app_data  # "Palette" or "Expressions"
        self.color_mode = "palette" if mode == "Palette" else "expressions"

        # Show/hide the appropriate UI groups
        dpg.configure_item("palette_mode_group", show=(self.color_mode == "palette"))
        dpg.configure_item("expressions_mode_group", show=(self.color_mode == "expressions"))

        if self.color_mode != "palette" and self.palette_editor_active:
            if self.palette_editor_dirty or self.palette_editor_persist_dirty:
                self._apply_palette_editor_refresh(persist=True)
            self._finalize_palette_editor_colormaps()
            self._set_palette_editor_state(False)
            self.palette_editor_refresh_pending = False

        # Update color_config based on mode
        with self.app.state_lock:
            if self.color_mode == "palette":
                # Clear color_config to use legacy palette mode
                self.app.state.color_config = None
            else:
                # Build ColorConfig from current expressions
                self._update_color_config_from_expressions()

        self.app.display_pipeline.refresh_display()

    def on_expression_preset_change(self, sender=None, app_data=None) -> None:
        """Handle preset selection change."""
        if dpg is None or app_data is None:
            return

        self._load_preset(app_data)

    def on_expression_change(self, sender=None, app_data=None) -> None:
        """Handle expression text change (debounced)."""
        if dpg is None:
            return

        # Mark pending update and record time
        self.expr_pending_update = True
        self.expr_last_update_time = time.time()

    def check_expression_debounce(self) -> None:
        """Check if expression update should be applied (called every frame)."""
        if not self.expr_pending_update:
            return

        current_time = time.time()
        if current_time - self.expr_last_update_time >= self.expr_debounce_delay:
            self._update_color_config_from_expressions()
            self.expr_pending_update = False

    def _update_color_config_from_expressions(self) -> None:
        """Build ColorConfig from current expression inputs and update state."""
        if dpg is None:
            return

        # Only update if in expressions mode
        if self.color_mode != "expressions":
            return

        from elliptica.colorspace import ColorConfig, ColorMapping
        from elliptica.expr import ExprError

        L_expr = dpg.get_value("expr_L_input").strip()
        C_expr = dpg.get_value("expr_C_input").strip()
        H_expr = dpg.get_value("expr_H_input").strip()

        # Try to build ColorConfig
        try:
            config = ColorConfig(
                global_mapping=ColorMapping(L=L_expr, C=C_expr, H=H_expr),
            )

            # Success - clear error and update state
            dpg.set_value("expr_error_text", "")

            with self.app.state_lock:
                self.app.state.color_config = config

            self.app.display_pipeline.refresh_display()

        except ExprError as e:
            # Show error but don't update config
            dpg.set_value("expr_error_text", f"Error: {e}")
