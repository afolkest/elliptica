"""Postprocessing panel controller for Elliptica UI - sliders, color, and region properties."""

from typing import Optional, TYPE_CHECKING

from elliptica import defaults
from elliptica.app.state_manager import StateKey
from elliptica.render import list_color_palettes
from elliptica.ui.dpg.expression_editor_mixin import ExpressionEditorMixin
from elliptica.ui.dpg.histogram_mixin import HistogramMixin
from elliptica.ui.dpg.palette_editor_mixin import PaletteEditorMixin

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class PostprocessingPanel(HistogramMixin, PaletteEditorMixin, ExpressionEditorMixin):
    """Controller for postprocessing sliders, color controls, and region properties.

    This class composes functionality from three mixins:
        - HistogramMixin: Histogram computation and rendering
        - PaletteEditorMixin: Gradient palette editing UI
        - ExpressionEditorMixin: OKLCH expression-based color mapping

    Mixin ordering matters: PaletteEditorMixin must precede ExpressionEditorMixin
    because ExpressionEditorMixin.on_color_mode_change() calls PaletteEditorMixin
    methods to finalize palette state when switching modes.

    Shared state initialized here before mixin inits:
        - palette_preview_width, palette_preview_height: Used by histogram and palette editor
        - color_mode: Shared between palette and expression editors
    """

    def __init__(self, app: "EllipticaApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main EllipticaApp instance
        """
        self.app = app

        # Palette preview UI dimensions (used by both histogram and palette editor mixins)
        self.palette_preview_width = 320
        self.palette_preview_height = 22

        # Initialize histogram state from mixin
        self._init_histogram_state()

        # Initialize palette editor state from mixin
        self._init_palette_editor_state()

        # Initialize expression editor state from mixin
        self._init_expression_editor_state()

        # Color mode: "palette" or "expressions" (shared by palette and expression editors)
        self.color_mode: str = "palette"

        # UI draft memory for lightness expressions (NOT state caches — these store
        # values intentionally removed from AppState so toggle on/off can restore them)
        self._cached_custom_lightness_exprs: dict[tuple[int, str], str] = {}
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

    def wire_subscribers(self) -> None:
        """Register StateManager subscribers (call after UI widgets exist)."""
        for key in (StateKey.SELECTED_INDICES, StateKey.SELECTED_REGION_TYPE):
            self.app.state_manager.subscribe(key, self._on_selection_changed)

    def _on_selection_changed(self, key, value, context) -> None:
        """Subscriber callback: refresh context UI when selection changes."""
        self.app.state_manager.flush_pending()
        self.update_context_ui()

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

                dpg.add_slider_float(
                    label="Clip low %",
                    default_value=self.app.state.display_settings.clip_low_percent,
                    min_value=0.0,
                    max_value=defaults.MAX_CLIP_PERCENT,
                    format="%.2f%%",
                    callback=self.on_clip_low_slider,
                    width=200,
                    tag="clip_low_slider",
                )
                dpg.add_slider_float(
                    label="Clip high %",
                    default_value=self.app.state.display_settings.clip_high_percent,
                    min_value=0.0,
                    max_value=defaults.MAX_CLIP_PERCENT,
                    format="%.2f%%",
                    callback=self.on_clip_high_slider,
                    width=200,
                    tag="clip_high_slider",
                )

                dpg.add_slider_float(
                    label="Brightness",
                    default_value=self.app.state.display_settings.brightness,
                    min_value=defaults.MIN_BRIGHTNESS,
                    max_value=defaults.MAX_BRIGHTNESS,
                    format="%.2f",
                    callback=self.on_brightness_slider,
                    width=200,
                    tag="brightness_slider",
                )
                dpg.add_slider_float(
                    label="Contrast",
                    default_value=self.app.state.display_settings.contrast,
                    min_value=defaults.MIN_CONTRAST,
                    max_value=defaults.MAX_CONTRAST,
                    format="%.2f",
                    callback=self.on_contrast_slider,
                    width=200,
                    tag="contrast_slider",
                )
                dpg.add_slider_float(
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
                    dpg.add_checkbox(
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
                    dpg.add_input_text(
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
            dpg.add_checkbox(
                label="Enable smear",
                callback=self.on_smear_enabled,
                tag="smear_enabled_checkbox",
            )
            dpg.add_slider_float(
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

    # ------------------------------------------------------------------
    # Region toggle callback
    # ------------------------------------------------------------------

    def on_region_toggle(self, sender=None, app_data=None) -> None:
        """Handle Surface/Interior region toggle."""
        if dpg is None:
            return

        # Flush pending debounced updates for the previous region before switching
        self.app.state_manager.flush_pending()
        region_type = "surface" if app_data == "Surface" else "interior"
        self.app.state_manager.update(StateKey.SELECTED_REGION_TYPE, region_type)
        # Subscribers handle: invalidate_selection_contour, update_context_ui

    # ------------------------------------------------------------------
    # Postprocessing slider callbacks
    # ------------------------------------------------------------------

    def on_clip_low_slider(self, sender=None, app_data=None) -> None:
        """Handle low clip slider change (debounced — percentile computation is expensive)."""
        if dpg is None:
            return

        self.app.state_manager.update(
            StateKey.CLIP_LOW_PERCENT, float(app_data), debounce=0.3,
        )
        self._request_histogram_update()

    def on_clip_high_slider(self, sender=None, app_data=None) -> None:
        """Handle high clip slider change (debounced — percentile computation is expensive)."""
        if dpg is None:
            return

        self.app.state_manager.update(
            StateKey.CLIP_HIGH_PERCENT, float(app_data), debounce=0.3,
        )
        self._request_histogram_update()

    def on_brightness_slider(self, sender=None, app_data=None) -> None:
        """Handle brightness slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        has_override, boundary_id, region_type = self._get_slider_context()

        if has_override and boundary_id is not None:
            self.app.state_manager.update(
                StateKey.REGION_STYLE, {"brightness": value},
                context=(boundary_id, region_type),
            )
        else:
            self.app.state_manager.update(StateKey.BRIGHTNESS, value)

        self._request_histogram_update()

    def on_contrast_slider(self, sender=None, app_data=None) -> None:
        """Handle contrast slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        has_override, boundary_id, region_type = self._get_slider_context()

        if has_override and boundary_id is not None:
            self.app.state_manager.update(
                StateKey.REGION_STYLE, {"contrast": value},
                context=(boundary_id, region_type),
            )
        else:
            self.app.state_manager.update(StateKey.CONTRAST, value)

        self._request_histogram_update()

    def on_gamma_slider(self, sender=None, app_data=None) -> None:
        """Handle gamma slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)
        has_override, boundary_id, region_type = self._get_slider_context()

        if has_override and boundary_id is not None:
            self.app.state_manager.update(
                StateKey.REGION_STYLE, {"gamma": value},
                context=(boundary_id, region_type),
            )
        else:
            self.app.state_manager.update(StateKey.GAMMA, value)

        self._request_histogram_update()

    def on_saturation_change(self, sender=None, app_data=None) -> None:
        """Handle saturation slider change (post-colorization chroma multiplier)."""
        if dpg is None:
            return

        self.app.state_manager.update(StateKey.SATURATION, float(app_data))

    # ------------------------------------------------------------------
    # Lightness expression callbacks (palette mode)
    # ------------------------------------------------------------------

    def on_lightness_expr_toggle(self, sender=None, app_data=None) -> None:
        """Handle lightness expression checkbox toggle (immediate, no debounce)."""
        if dpg is None:
            return

        is_enabled = bool(app_data)
        dpg.configure_item("lightness_expr_group", show=is_enabled)

        if is_enabled:
            expr = self._cached_global_lightness_expr or "clipnorm(mag, 1, 99)"
            dpg.set_value("lightness_expr_input", expr)
            self.app.state_manager.update(StateKey.LIGHTNESS_EXPR, expr)
        else:
            with self.app.state_lock:
                current = self.app.state.display_settings.lightness_expr
            if current:
                self._cached_global_lightness_expr = current
            self.app.state_manager.update(StateKey.LIGHTNESS_EXPR, None)

    def on_lightness_expr_change(self, sender=None, app_data=None) -> None:
        """Handle lightness expression text change (debounced via StateManager)."""
        if dpg is None:
            return

        expr = dpg.get_value("lightness_expr_input").strip()
        if not expr:
            return

        # Route based on UI state captured NOW (radio button reflects user intent)
        if self._is_boundary_selected():
            mode = dpg.get_value("lightness_expr_mode_radio")
            if mode == "Custom":
                with self.app.state_lock:
                    selected = self.app.state.get_selected()
                    region_type = self.app.state.selected_region_type
                if selected and selected.id is not None:
                    cache_key = (selected.id, region_type)
                    self._cached_custom_lightness_exprs[cache_key] = expr
                    self.app.state_manager.update(
                        StateKey.REGION_STYLE,
                        {"lightness_expr": expr},
                        context=(selected.id, region_type),
                        debounce=0.3,
                    )
                    return

        # Global update (no boundary, or Global mode, or fallback)
        self._cached_global_lightness_expr = expr
        self.app.state_manager.update(StateKey.LIGHTNESS_EXPR, expr, debounce=0.3)

    # ------------------------------------------------------------------
    # Lightness expression mode (Global/Custom) callback
    # ------------------------------------------------------------------

    def on_lightness_expr_mode_change(self, sender=None, app_data=None) -> None:
        """Handle Global/Custom radio toggle for per-region lightness expression."""
        if dpg is None:
            return

        # Flush any pending debounced lightness expr updates before mode change
        self.app.state_manager.flush_pending()

        is_custom = (app_data == "Custom")

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected is None or selected.id is None:
                return
            region_type = self.app.state.selected_region_type

        cache_key = (selected.id, region_type)

        if is_custom:
            # Switch to custom - check cache first, then global, then default
            cached_expr = self._cached_custom_lightness_exprs.get(cache_key)
            if cached_expr is None:
                with self.app.state_lock:
                    cached_expr = self.app.state.display_settings.lightness_expr or "clipnorm(mag, 1, 99)"
            self.app.state_manager.update(
                StateKey.REGION_STYLE,
                {"lightness_expr": cached_expr},
                context=(selected.id, region_type),
            )
        else:
            # Switch to global - cache current custom expr before clearing
            with self.app.state_lock:
                from elliptica.app.core import BoundaryColorSettings
                bcs = self.app.state.boundary_color_settings.get(selected.id)
                if bcs is not None:
                    rs = bcs.surface if region_type == "surface" else bcs.interior
                    if rs.lightness_expr is not None:
                        self._cached_custom_lightness_exprs[cache_key] = rs.lightness_expr
            self.app.state_manager.update(
                StateKey.REGION_STYLE,
                {"lightness_expr": None},
                context=(selected.id, region_type),
            )

        self.update_context_ui()

    # ------------------------------------------------------------------
    # Smear callbacks
    # ------------------------------------------------------------------

    def on_smear_enabled(self, sender=None, app_data=None) -> None:
        """Toggle smear for selected region."""
        if dpg is None:
            return

        selected = self.app.state.get_selected()
        if selected is None or selected.id is None:
            return
        context = (selected.id, self.app.state.selected_region_type)
        self.app.state_manager.update(
            StateKey.REGION_STYLE,
            {"smear_enabled": bool(app_data)},
            context=context,
        )
        self.update_context_ui()

    def on_smear_sigma(self, sender=None, app_data=None) -> None:
        """Adjust smear blur sigma for selected region (debounced)."""
        if dpg is None:
            return

        selected = self.app.state.get_selected()
        if selected is None or selected.id is None:
            return
        context = (selected.id, self.app.state.selected_region_type)
        self.app.state_manager.update(
            StateKey.REGION_STYLE,
            {"smear_sigma": float(app_data)},
            debounce=0.3,
            context=context,
        )

