"""Postprocessing panel controller for Elliptica UI - sliders, color, and region properties."""

import time
import numpy as np
from PIL import Image
from typing import Optional, Literal, TYPE_CHECKING

from elliptica import defaults
from elliptica.app import actions
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
        self.hist_values: Optional[np.ndarray] = None
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
        self.palette_editor_preview_button_id: Optional[int] = None
        self.palette_editor_info_text_id: Optional[int] = None
        self.palette_editor_l_slider_id: Optional[int] = None
        self.palette_editor_c_slider_id: Optional[int] = None
        self.palette_editor_h_slider_id: Optional[int] = None

        self.palette_editor_width = self.palette_preview_width
        self.palette_editor_gradient_height = 60
        self.palette_editor_slice_height = 110
        self.palette_editor_lbar_height = 16
        self.palette_editor_preview_size = 60

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
                    dpg.add_text("Lightness expr", color=(150, 150, 150), tag="lightness_expr_label")
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
            self.palette_editor_l_gradient_drawlist_id = dpg.add_drawlist(
                width=self.palette_editor_width,
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
                width=self.palette_editor_width,
                enabled=False,
                tag="palette_editor_l_slider",
            )
            self.palette_editor_c_slider_id = dpg.add_slider_float(
                label="C",
                default_value=0.0,
                min_value=0.0,
                max_value=0.4,
                format="%.4f",
                width=self.palette_editor_width,
                enabled=False,
                tag="palette_editor_c_slider",
            )
            self.palette_editor_h_slider_id = dpg.add_slider_float(
                label="H",
                default_value=0.0,
                min_value=0.0,
                max_value=360.0,
                format="%.1f",
                width=self.palette_editor_width,
                enabled=False,
                tag="palette_editor_h_slider",
            )

            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                self.palette_editor_preview_button_id = dpg.add_color_button(
                    default_value=(0.3, 0.3, 0.3, 1.0),
                    width=self.palette_editor_preview_size,
                    height=self.palette_editor_preview_size,
                    enabled=False,
                    tag="palette_editor_preview",
                )
                dpg.add_spacer(width=8)
                self.palette_editor_info_text_id = dpg.add_text(
                    "Select a stop to edit.",
                    tag="palette_editor_info_text",
                    color=(140, 140, 140),
                )

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
    def _build_expression_editor_ui(self, parent) -> None:
        """Build the expression editor UI for OKLCH color mapping.

        Args:
            parent: Parent widget ID
        """
        if dpg is None:
            return

        from elliptica.colorspace import list_presets, get_preset, AVAILABLE_VARIABLES, AVAILABLE_FUNCTIONS, PDE_SPECIFIC_VARIABLES

        preset_names = list_presets()

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

        # L expression
        dpg.add_text("Lightness (L)  [0-1]", parent=parent)
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

        # C expression
        dpg.add_text("Chroma (C)  [0-0.4]", parent=parent)
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

        # H expression
        dpg.add_text("Hue (H)  [0-360°]", parent=parent)
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

    def _compute_histogram_values(self) -> Optional[np.ndarray]:
        """Compute histogram for post-processed intensity (global clip/contrast/gamma)."""
        with self.app.state_lock:
            cache = self.app.state.render_cache
            if cache is None or cache.result is None:
                return None

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

        sample = self._downsample_histogram_source(source)

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

        counts, _ = np.histogram(adjusted, bins=self.palette_hist_bins, range=(0.0, 1.0))
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

        self.hist_values = self._compute_histogram_values()
        self._update_histogram_drawlist(self.global_hist_drawlist_id, self.hist_values)
        self._update_histogram_drawlist(self.region_hist_drawlist_id, self.hist_values)
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
        # Cancel ALL pending debounced updates - context is changing
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
        if not has_override:
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
        if not has_override:
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
        if not has_override:
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
        popup_tag = "region_palette_popup" if for_region else "global_palette_popup"
        if dpg.does_item_exist(popup_tag):
            dpg.configure_item(popup_tag, show=False)

    def on_palette_editor_done(self, sender=None, app_data=None) -> None:
        """Exit palette edit mode."""
        if dpg is None:
            return
        self._set_palette_editor_state(False)

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
            self._set_palette_editor_state(False)

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
