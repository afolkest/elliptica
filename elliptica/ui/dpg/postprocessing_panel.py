"""Postprocessing panel controller for Elliptica UI - sliders, color, and region properties."""

import time
from typing import Optional, Literal, TYPE_CHECKING

from elliptica import defaults
from elliptica.app import actions

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
        self.postprocess_clip_slider_id: Optional[int] = None
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

        # Color mode: "palette" or "expressions"
        self.color_mode: str = "palette"

        # Region editing context: which region we're editing when conductor selected
        self.selected_region: Literal["surface", "interior"] = "surface"

        # Delete confirmation modal
        self.delete_confirmation_modal_id: Optional[int] = None
        self.pending_delete_palette: Optional[str] = None

        # Debouncing for expensive smear updates
        self.smear_pending_value: Optional[float] = None
        self.smear_last_update_time: float = 0.0
        self.smear_debounce_delay: float = 0.3  # 300ms delay

        # Debouncing for expensive clip% updates (percentile computation at high res)
        self.clip_pending_value: Optional[float] = None
        self.clip_last_update_time: float = 0.0
        self.clip_debounce_delay: float = 0.3  # 300ms delay

        # Debouncing for expression updates
        self.expr_pending_update: bool = False
        self.expr_last_update_time: float = 0.0
        self.expr_debounce_delay: float = 0.3  # 300ms delay

        # Themes for grayed-out appearance
        self.disabled_theme_id: Optional[int] = None
        self.disabled_slider_theme_id: Optional[int] = None
        self.disabled_button_theme_id: Optional[int] = None

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

    def _is_conductor_selected(self) -> bool:
        """Check if a conductor is currently selected."""
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            return selected is not None and selected.id is not None

    def _get_current_region_style(self):
        """Get the RegionStyle for the currently selected context, or None if global."""
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected is None or selected.id is None:
                return None
            settings = self.app.state.conductor_color_settings.get(selected.id)
            if settings is None:
                return None
            if self.selected_region == "surface":
                return settings.surface
            else:
                return settings.interior

    def _has_override_enabled(self) -> bool:
        """Check if the current region has postprocessing override enabled."""
        region_style = self._get_current_region_style()
        if region_style is None:
            return False
        # Override is enabled if any of brightness/contrast/gamma is set
        return (region_style.brightness is not None or
                region_style.contrast is not None or
                region_style.gamma is not None)

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

        # Color mode toggle
        with dpg.group(horizontal=True, parent=parent):
            dpg.add_text("Color Mode:", color=LABEL_TEXT)
            dpg.add_radio_button(
                items=["Palette", "Expressions"],
                default_value="Palette",
                horizontal=True,
                callback=self.on_color_mode_change,
                tag="color_mode_radio",
            )

        dpg.add_spacer(height=10, parent=parent)

        # Palette mode container (shown by default)
        with dpg.group(tag="palette_mode_group", parent=parent):
            # === FIXED CONTEXT AREA (always takes same vertical space) ===
            # Context header - ALWAYS visible, text changes based on mode
            dpg.add_text("Global Settings", tag="context_header_text", color=(150, 200, 255))

            # Region controls line - ONLY shown when conductor selected
            with dpg.group(horizontal=True, tag="region_controls_line", show=False):
                dpg.add_radio_button(
                    items=["Surface", "Interior"],
                    default_value="Surface",
                    horizontal=True,
                    callback=self.on_region_toggle,
                    tag="region_toggle_radio",
                )
                dpg.add_spacer(width=10)
                dpg.add_checkbox(
                    label="Override",
                    callback=self.on_enable_override,
                    tag="enable_override_checkbox",
                )

            # Placeholder spacer - shown in global mode to reserve same vertical space
            # This prevents sliders from jumping when switching between modes
            dpg.add_spacer(height=22, tag="region_controls_placeholder", show=True)

            dpg.add_spacer(height=5)

            # === PALETTE AREA ===
            # Global palette UI (shown in global mode)
            with dpg.group(tag="global_palette_group"):
                self._build_global_palette_ui("global_palette_group", palette_colormaps)

            # Region palette UI (shown in conductor mode)
            with dpg.group(tag="region_palette_group", show=False):
                self._build_region_palette_ui("region_palette_group", palette_colormaps)

            # === SLIDERS (fixed position, never move) ===
            dpg.add_spacer(height=10)

            # Clip% - always shows global, disabled in conductor mode
            self.postprocess_clip_slider_id = dpg.add_slider_float(
                label="Clip % (global)",
                default_value=self.app.state.display_settings.clip_percent,
                min_value=0.0,
                max_value=defaults.MAX_CLIP_PERCENT,
                format="%.2f%%",
                callback=self.on_clip_slider,
                width=200,
                tag="clip_slider",
            )

            # Brightness/Contrast/Gamma sliders
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

        # Expressions mode container (hidden by default)
        with dpg.group(tag="expressions_mode_group", parent=parent, show=False):
            self._build_expression_editor_ui("expressions_mode_group")

        dpg.add_spacer(height=10, parent=parent)
        dpg.add_separator(parent=parent)

        # Conductor effects section (smear) - only shown when conductor selected
        with dpg.collapsing_header(label="Conductor Effects", default_open=True,
                                   tag="conductor_effects_header", parent=parent, show=False):
            dpg.add_text("Smear")
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

        from elliptica.render import list_color_palettes
        palette_names = list(list_color_palettes())

        # Palette label and button showing current selection
        LABEL_TEXT = (150, 150, 150)
        dpg.add_text("Palette", parent=parent, color=LABEL_TEXT)

        # Button shows current palette name directly
        initial_label = self.app.state.display_settings.palette if self.app.state.display_settings.color_enabled else "Grayscale"
        global_palette_button = dpg.add_button(
            label=initial_label,
            width=200,
            tag="global_palette_button",
            parent=parent,
        )

        # Popup menu for global palette selection
        with dpg.popup(global_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="global_palette_popup"):
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

    def _build_region_palette_ui(self, parent, palette_colormaps: dict) -> None:
        """Build region palette selection UI with popup menu.

        Args:
            parent: Parent widget ID
            palette_colormaps: Dict mapping palette names to colormap tags
        """
        if dpg is None:
            return

        from elliptica.render import list_color_palettes
        palette_names = list(list_color_palettes())

        # Region palette label and button showing current selection
        LABEL_TEXT = (150, 150, 150)
        dpg.add_text("Region Palette", parent=parent, color=LABEL_TEXT)

        # Button shows current selection directly
        region_palette_button = dpg.add_button(
            label="Use Global",
            width=200,
            tag="region_palette_button",
            parent=parent,
        )

        # Popup menu for region palette selection
        with dpg.popup(region_palette_button, mousebutton=dpg.mvMouseButton_Left, tag="region_palette_popup"):
            dpg.add_text("Select region palette:")
            dpg.add_separator()

            # Add "Use Global Palette" option at top
            dpg.add_button(
                label="⊙ Use Global Palette",
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

    def _build_expression_editor_ui(self, parent) -> None:
        """Build the expression editor UI for OKLCH color mapping.

        Args:
            parent: Parent widget ID
        """
        if dpg is None:
            return

        from elliptica.colorspace import list_presets, get_preset, AVAILABLE_VARIABLES, AVAILABLE_FUNCTIONS

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

            dpg.add_spacer(height=8)
            dpg.add_text("Functions:", color=(150, 200, 255))
            for func_sig, func_desc in AVAILABLE_FUNCTIONS:
                dpg.add_text(f"  {func_sig}", color=(200, 200, 200))
                dpg.add_text(f"    {func_desc}", color=(150, 150, 150), wrap=260)

        # Load first preset
        if preset_names:
            self._load_preset(preset_names[0])

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

    def update_context_ui(self) -> None:
        """Update UI based on current selection context (global vs conductor)."""
        if dpg is None:
            return

        is_conductor_selected = self._is_conductor_selected()

        if is_conductor_selected:
            # Conductor mode
            with self.app.state_lock:
                selected = self.app.state.get_selected()
                conductor_idx = self.app.state.selected_idx

            # Update context header text (always visible, just change text)
            region_label = "Surface" if self.selected_region == "surface" else "Interior"
            dpg.set_value("context_header_text", f"Conductor {conductor_idx + 1} - {region_label}")

            # Show region controls, hide placeholder
            dpg.configure_item("region_controls_line", show=True)
            dpg.configure_item("region_controls_placeholder", show=False)

            # Switch palette UI
            dpg.configure_item("global_palette_group", show=False)
            dpg.configure_item("region_palette_group", show=True)

            # Check if override is enabled
            has_override = self._has_override_enabled()
            dpg.set_value("enable_override_checkbox", has_override)

            # Update region palette button label and grayed state
            region_style = self._get_current_region_style()
            if region_style and region_style.enabled:
                dpg.configure_item("region_palette_button", label=region_style.palette)
            else:
                dpg.configure_item("region_palette_button", label="Use Global")
            self._set_button_grayed("region_palette_button", not has_override)

            # Clip% always shows global, grayed in conductor mode
            self._set_slider_grayed("clip_slider", True)
            dpg.set_value("clip_slider", self.app.state.display_settings.clip_percent)

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

            # Show conductor effects section
            dpg.configure_item("conductor_effects_header", show=True)

            # Update smear controls
            dpg.set_value("smear_enabled_checkbox", selected.smear_enabled)
            dpg.set_value("smear_sigma_slider", selected.smear_sigma)
            dpg.configure_item("smear_sigma_slider", show=selected.smear_enabled)

        else:
            # Global mode
            dpg.set_value("context_header_text", "Global Settings")

            # Hide region controls, show placeholder (maintains layout)
            dpg.configure_item("region_controls_line", show=False)
            dpg.configure_item("region_controls_placeholder", show=True)

            # Switch palette UI
            dpg.configure_item("global_palette_group", show=True)
            dpg.configure_item("region_palette_group", show=False)

            # Clip% normal in global mode
            self._set_slider_grayed("clip_slider", False)
            dpg.set_value("clip_slider", self.app.state.display_settings.clip_percent)

            # B/C/G sliders normal, showing global values
            self._set_slider_grayed("brightness_slider", False)
            self._set_slider_grayed("contrast_slider", False)
            self._set_slider_grayed("gamma_slider", False)
            dpg.set_value("brightness_slider", self.app.state.display_settings.brightness)
            dpg.set_value("contrast_slider", self.app.state.display_settings.contrast)
            dpg.set_value("gamma_slider", self.app.state.display_settings.gamma)

            # Hide conductor effects section
            dpg.configure_item("conductor_effects_header", show=False)

    def update_region_properties_panel(self) -> None:
        """Update region properties panel based on current selection.

        This is called when selection changes. Delegates to update_context_ui.
        """
        self.update_context_ui()

    # ------------------------------------------------------------------
    # Region toggle callback
    # ------------------------------------------------------------------

    def on_region_toggle(self, sender=None, app_data=None) -> None:
        """Handle Surface/Interior region toggle."""
        if dpg is None:
            return

        self.selected_region = "surface" if app_data == "Surface" else "interior"
        self.update_context_ui()

    # ------------------------------------------------------------------
    # Enable Override callback
    # ------------------------------------------------------------------

    def on_enable_override(self, sender=None, app_data=None) -> None:
        """Toggle postprocessing override for current region."""
        if dpg is None:
            return

        is_enabled = bool(app_data)

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                if is_enabled:
                    # Initialize with global values
                    global_brightness = self.app.state.display_settings.brightness
                    global_contrast = self.app.state.display_settings.contrast
                    global_gamma = self.app.state.display_settings.gamma
                    actions.set_region_brightness(self.app.state, selected.id, self.selected_region, global_brightness)
                    actions.set_region_contrast(self.app.state, selected.id, self.selected_region, global_contrast)
                    actions.set_region_gamma(self.app.state, selected.id, self.selected_region, global_gamma)
                else:
                    # Reset to None (inherit from global)
                    actions.set_region_brightness(self.app.state, selected.id, self.selected_region, None)
                    actions.set_region_contrast(self.app.state, selected.id, self.selected_region, None)
                    actions.set_region_gamma(self.app.state, selected.id, self.selected_region, None)

        self.update_context_ui()
        self.app.display_pipeline.refresh_display()

    # ------------------------------------------------------------------
    # Postprocessing slider callbacks
    # ------------------------------------------------------------------

    def on_clip_slider(self, sender=None, app_data=None) -> None:
        """Handle clip percent slider change with debouncing (percentile computation is expensive at high res)."""
        if dpg is None:
            return

        # Clip% always edits global (disabled in conductor mode anyway)
        value = float(app_data)

        # IMMEDIATELY update state so other refreshes use the correct value
        with self.app.state_lock:
            self.app.state.display_settings.clip_percent = value
            self.app.state.invalidate_base_rgb()

        # Mark pending to trigger refresh after debounce delay
        self.clip_pending_value = value
        self.clip_last_update_time = time.time()

    def on_brightness_slider(self, sender=None, app_data=None) -> None:
        """Handle brightness slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)

        if self._is_conductor_selected() and self._has_override_enabled():
            # Per-region brightness
            with self.app.state_lock:
                selected = self.app.state.get_selected()
                if selected and selected.id is not None:
                    actions.set_region_brightness(self.app.state, selected.id, self.selected_region, value)
        else:
            # Global brightness
            with self.app.state_lock:
                self.app.state.display_settings.brightness = value
                self.app.state.invalidate_base_rgb()

        self.app.display_pipeline.refresh_display()

    def on_contrast_slider(self, sender=None, app_data=None) -> None:
        """Handle contrast slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)

        if self._is_conductor_selected() and self._has_override_enabled():
            # Per-region contrast
            with self.app.state_lock:
                selected = self.app.state.get_selected()
                if selected and selected.id is not None:
                    actions.set_region_contrast(self.app.state, selected.id, self.selected_region, value)
        else:
            # Global contrast
            with self.app.state_lock:
                self.app.state.display_settings.contrast = value
                self.app.state.invalidate_base_rgb()

        self.app.display_pipeline.refresh_display()

    def on_gamma_slider(self, sender=None, app_data=None) -> None:
        """Handle gamma slider change (real-time with GPU acceleration)."""
        if dpg is None:
            return

        value = float(app_data)

        if self._is_conductor_selected() and self._has_override_enabled():
            # Per-region gamma
            with self.app.state_lock:
                selected = self.app.state.get_selected()
                if selected and selected.id is not None:
                    actions.set_region_gamma(self.app.state, selected.id, self.selected_region, value)
        else:
            # Global gamma
            with self.app.state_lock:
                self.app.state.display_settings.gamma = value
                self.app.state.invalidate_base_rgb()

        self.app.display_pipeline.refresh_display()

    # ------------------------------------------------------------------
    # Color and palette callbacks
    # ------------------------------------------------------------------

    def on_global_grayscale(self, sender=None, app_data=None) -> None:
        """Handle global 'Grayscale (No Color)' button."""
        if dpg is None:
            return

        with self.app.state_lock:
            actions.set_color_enabled(self.app.state, False)

        # Update button label to show current selection
        dpg.configure_item("global_palette_button", label="Grayscale")
        dpg.configure_item("global_palette_popup", show=False)

        self.app.display_pipeline.refresh_display()

    def on_global_palette_button(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle global colormap button click."""
        if dpg is None or user_data is None:
            return

        palette_name = user_data
        with self.app.state_lock:
            # Auto-enable color when a palette is selected
            actions.set_color_enabled(self.app.state, True)
            actions.set_palette(self.app.state, palette_name)

        # Update button label to show current selection
        dpg.configure_item("global_palette_button", label=palette_name)
        dpg.configure_item("global_palette_popup", show=False)

        self.app.display_pipeline.refresh_display()

    def on_region_use_global(self, sender=None, app_data=None) -> None:
        """Handle region 'Use Global Palette' button."""
        if dpg is None:
            return

        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_style_enabled(self.app.state, selected.id, self.selected_region, False)

        # Update button label to show current selection
        dpg.configure_item("region_palette_button", label="Use Global")
        dpg.configure_item("region_palette_popup", show=False)

        self.app.display_pipeline.refresh_display()

    def on_region_palette_button(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle region colormap button click."""
        if dpg is None or user_data is None:
            return

        palette_name = user_data
        with self.app.state_lock:
            selected = self.app.state.get_selected()
            if selected and selected.id is not None:
                actions.set_region_palette(self.app.state, selected.id, self.selected_region, palette_name)

        # Update button label to show current selection
        dpg.configure_item("region_palette_button", label=palette_name)
        dpg.configure_item("region_palette_popup", show=False)

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

        from elliptica.render import delete_palette

        palette_name = self.pending_delete_palette
        delete_palette(palette_name)

        # Rebuild DPG colormaps to reflect deletion
        self.app.display_pipeline.texture_manager.rebuild_colormaps()

        # Rebuild the palette popup menu
        self._rebuild_palette_popup()

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
        from elliptica.render import list_color_palettes
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

    # ------------------------------------------------------------------
    # Smear callbacks
    # ------------------------------------------------------------------

    def on_smear_enabled(self, sender=None, app_data=None) -> None:
        """Toggle interior smear for selected conductor region."""
        if dpg is None:
            return

        with self.app.state_lock:
            idx = self.app.state.selected_idx
            if idx >= 0 and idx < len(self.app.state.project.conductors):
                self.app.state.project.conductors[idx].smear_enabled = bool(app_data)

        self.update_context_ui()
        self.app.display_pipeline.refresh_display()

    def on_smear_sigma(self, sender=None, app_data=None) -> None:
        """Adjust smear blur sigma for selected conductor (debounced for performance)."""
        if dpg is None:
            return

        # Always update the value immediately (for UI responsiveness)
        with self.app.state_lock:
            idx = self.app.state.selected_idx
            if idx >= 0 and idx < len(self.app.state.project.conductors):
                self.app.state.project.conductors[idx].smear_sigma = float(app_data)

        # Mark that we have a pending update (defers expensive render refresh)
        self.smear_pending_value = float(app_data)

        # Record the time of THIS slider change (not the last render)
        self.smear_last_update_time = time.time()

    def _apply_clip_update(self, value: float) -> None:
        """Apply clip percent refresh (state already updated in on_clip_slider)."""
        self.app.display_pipeline.refresh_display()

    def _apply_smear_update(self) -> None:
        """Apply pending smear update (called after debounce delay)."""
        if self.smear_pending_value is None:
            return

        self.smear_pending_value = None
        self.app.display_pipeline.refresh_display()

    def check_clip_debounce(self) -> None:
        """Check if clip update should be applied (called every frame)."""
        if self.clip_pending_value is None:
            return

        current_time = time.time()
        # Only apply if enough time has passed since the last slider movement
        if current_time - self.clip_last_update_time >= self.clip_debounce_delay:
            self._apply_clip_update(self.clip_pending_value)
            self.clip_last_update_time = current_time
            self.clip_pending_value = None

    def check_smear_debounce(self) -> None:
        """Check if smear update should be applied (called every frame)."""
        if self.smear_pending_value is None:
            return

        current_time = time.time()
        # Only apply if enough time has passed since the last slider movement
        if current_time - self.smear_last_update_time >= self.smear_debounce_delay:
            self._apply_smear_update()

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
