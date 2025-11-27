"""Render modal dialog controller for FlowCol UI."""

from dataclasses import replace
from typing import Optional, TYPE_CHECKING

from flowcol import defaults
from flowcol.app import actions
from flowcol.pde.boundary_utils import resolve_bc_map, bc_map_to_legacy

if TYPE_CHECKING:
    from flowcol.ui.dpg.app import FlowColApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


# Constants for render settings
SUPERSAMPLE_CHOICES = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
SUPERSAMPLE_LABELS = ["1× (fastest)", "1.25×", "1.5×", "2×", "3×", "4× (highest quality)"]
SUPERSAMPLE_LOOKUP = dict(zip(SUPERSAMPLE_LABELS, SUPERSAMPLE_CHOICES))

RESOLUTION_CHOICES = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
RESOLUTION_LABELS = ["1× (canvas)", "1.5×", "2×", "3×", "4×", "6×", "8×"]
RESOLUTION_LOOKUP = dict(zip(RESOLUTION_LABELS, RESOLUTION_CHOICES))


def _label_for_supersample(value: float) -> str:
    """Get radio button label for supersample value."""
    try:
        idx = SUPERSAMPLE_CHOICES.index(value)
        return SUPERSAMPLE_LABELS[idx]
    except (ValueError, IndexError):
        return SUPERSAMPLE_LABELS[0]


def _label_for_multiplier(value: float) -> str:
    """Get radio button label for multiplier value."""
    try:
        idx = RESOLUTION_CHOICES.index(value)
        return RESOLUTION_LABELS[idx]
    except (ValueError, IndexError):
        return RESOLUTION_LABELS[0]


class RenderModalController:
    """Controller for render settings modal dialog."""

    def __init__(self, app: "FlowColApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main FlowColApp instance
        """
        self.app = app

        # Modal window and state
        self.modal_id: Optional[int] = None
        self.modal_open: bool = False

        # Widget IDs for form controls
        self.supersample_radio_id: Optional[int] = None
        self.multiplier_radio_id: Optional[int] = None
        self.passes_input_id: Optional[int] = None
        self.streamlength_input_id: Optional[int] = None
        self.margin_input_id: Optional[int] = None
        self.seed_input_id: Optional[int] = None
        self.sigma_input_id: Optional[int] = None
        self.solve_scale_slider_id: Optional[int] = None
        self.use_mask_checkbox_id: Optional[int] = None
        self.edge_gain_strength_slider_id: Optional[int] = None
        self.edge_gain_power_slider_id: Optional[int] = None

        # Generic Boundary Condition Controls
        # Map: edge_name -> field_name -> widget_id
        self.bc_field_ids: dict[str, dict[str, int]] = {}
        # Enum choices cache: edge -> field_name -> [(label, value)]
        self.bc_enum_choices: dict[str, dict[str, list[tuple[str, object]]]] = {}
        # Field metadata cache by name
        self.bc_field_defs: dict[str, object] = {}

    def ensure_modal(self) -> None:
        """Create the modal dialog window if it doesn't exist."""
        if dpg is None or self.modal_id is not None:
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
            height=550,
        ) as modal:
            self.modal_id = modal

            dpg.add_text("Supersample Factor")
            self.supersample_radio_id = dpg.add_radio_button(
                SUPERSAMPLE_LABELS,
                horizontal=True,
            )
            dpg.add_spacer(height=10)

            dpg.add_text("Render Resolution")
            self.multiplier_radio_id = dpg.add_radio_button(
                RESOLUTION_LABELS,
                horizontal=True,
            )
            dpg.add_spacer(height=12)

            dpg.add_separator()
            dpg.add_spacer(height=12)

            self.passes_input_id = dpg.add_input_int(
                label="LIC Passes",
                min_value=1,
                step=1,
                min_clamped=True,
                width=160,
            )

            self.streamlength_input_id = dpg.add_input_float(
                label="Streamlength Factor",
                format="%.4f",
                min_value=1e-6,
                min_clamped=True,
                step=0.0,
                width=200,
            )

            self.margin_input_id = dpg.add_input_float(
                label="Padding Margin",
                format="%.3f",
                min_value=0.0,
                min_clamped=True,
                step=0.0,
                width=200,
            )

            dpg.add_spacer(height=8)
            self.solve_scale_slider_id = dpg.add_slider_float(
                label="Solve Scale",
                default_value=defaults.DEFAULT_SOLVE_SCALE,
                min_value=defaults.MIN_SOLVE_SCALE,
                max_value=defaults.MAX_SOLVE_SCALE,
                format="%.2f",
                width=250,
            )
            dpg.add_text("PDE grid resolution (lower = faster, upsampled to render size).")

            self.seed_input_id = dpg.add_input_int(
                label="Noise Seed",
                step=1,
                min_clamped=False,
                width=160,
            )

            self.sigma_input_id = dpg.add_input_float(
                label="Noise Low-pass Sigma",
                format="%.2f",
                min_value=0.0,
                min_clamped=True,
                step=0.0,
                width=200,
            )

            dpg.add_spacer(height=10)
            self.use_mask_checkbox_id = dpg.add_checkbox(
                label="Block streamlines at conductors",
                default_value=defaults.DEFAULT_USE_MASK,
            )

            dpg.add_spacer(height=8)
            self.edge_gain_strength_slider_id = dpg.add_slider_float(
                label="Edge Halo Strength",
                default_value=defaults.DEFAULT_EDGE_GAIN_STRENGTH,
                min_value=defaults.MIN_EDGE_GAIN_STRENGTH,
                max_value=defaults.MAX_EDGE_GAIN_STRENGTH,
                format="%.2f",
                width=250,
            )

            self.edge_gain_power_slider_id = dpg.add_slider_float(
                label="Edge Halo Power",
                default_value=defaults.DEFAULT_EDGE_GAIN_POWER,
                min_value=defaults.MIN_EDGE_GAIN_POWER,
                max_value=defaults.MAX_EDGE_GAIN_POWER,
                format="%.2f",
                width=250,
            )

            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            # Dynamic Boundary Conditions Section (per-PDE)
            from flowcol.pde import PDERegistry
            pde = PDERegistry.get_active()
            self._build_bc_controls(modal, pde)

            dpg.add_spacer(height=20)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Render", width=140, callback=self.on_apply)
                dpg.add_button(label="Cancel", width=140, callback=self.on_cancel)

    def _build_bc_controls(self, parent, pde) -> None:
        """Build boundary condition controls from PDE metadata."""
        if dpg is None:
            return

        self.bc_field_ids = {}
        self.bc_enum_choices = {}
        self.bc_field_defs = {f.name: f for f in getattr(pde, "bc_fields", [])}

        bc_fields = getattr(pde, "bc_fields", [])
        if not bc_fields:
            dpg.add_text("No global boundary conditions available.", parent=parent)
            return

        dpg.add_text("Boundary Conditions")
        dpg.add_spacer(height=5)

        edges = ["top", "right", "bottom", "left"]
        for edge in edges:
            self.bc_field_ids[edge] = {}
            self.bc_enum_choices[edge] = {}
            with dpg.collapsing_header(label=edge.title(), default_open=True, parent=parent):
                for field in bc_fields:
                    if field.field_type == "enum":
                        labels = [lbl for lbl, _ in field.choices]
                        self.bc_enum_choices[edge][field.name] = list(field.choices)
                        widget_id = dpg.add_combo(
                            label=field.display_name,
                            items=labels,
                            width=200,
                            callback=self._on_bc_type_changed,
                            user_data=edge,
                        )
                    elif field.field_type == "bool":
                        widget_id = dpg.add_checkbox(
                            label=field.display_name,
                        )
                    elif field.field_type == "int":
                        widget_id = dpg.add_input_int(
                            label=field.display_name,
                            width=120,
                            min_value=field.min_value if field.min_value is not None else 0,
                            max_value=field.max_value if field.max_value is not None else 2147483647,
                        )
                    else:
                        widget_id = dpg.add_input_float(
                            label=field.display_name,
                            width=180,
                            min_value=field.min_value if field.min_value is not None else 0.0,
                            max_value=field.max_value if field.max_value is not None else 1e9,
                        )

                    self.bc_field_ids[edge][field.name] = widget_id
                    if getattr(field, "description", ""):
                        with dpg.tooltip(widget_id):
                            dpg.add_text(field.description)

        # Set initial visibility based on default BC type (Dirichlet)
        self._update_bc_field_visibility()

    def _on_bc_type_changed(self, sender, app_data, user_data) -> None:
        """Callback when any BC enum field changes - updates dependent field visibility."""
        self._update_bc_field_visibility()

    def _update_bc_field_visibility(self) -> None:
        """Show/hide BC fields based on their visible_when rules."""
        if dpg is None:
            return

        for edge, field_ids in self.bc_field_ids.items():
            # First, collect current values of all fields for this edge
            current_values = {}
            for field_name, widget_id in field_ids.items():
                field_def = self.bc_field_defs.get(field_name)
                if field_def is None or not dpg.does_item_exist(widget_id):
                    continue
                if field_def.field_type == "enum":
                    # Convert label back to value
                    label = dpg.get_value(widget_id)
                    choices = self.bc_enum_choices.get(edge, {}).get(field_name, [])
                    value = None
                    for lbl, v in choices:
                        if lbl == label:
                            value = v
                            break
                    current_values[field_name] = value
                else:
                    current_values[field_name] = dpg.get_value(widget_id)

            # Now apply visibility rules to each field
            for field_name, widget_id in field_ids.items():
                field_def = self.bc_field_defs.get(field_name)
                if field_def is None or not dpg.does_item_exist(widget_id):
                    continue

                visible_when = getattr(field_def, "visible_when", None)
                if visible_when is None:
                    # No rule means always visible
                    dpg.configure_item(widget_id, show=True)
                else:
                    # Check all conditions in visible_when dict
                    should_show = True
                    for dep_field, required_value in visible_when.items():
                        if current_values.get(dep_field) != required_value:
                            should_show = False
                            break
                    dpg.configure_item(widget_id, show=should_show)

    def open(self, sender=None, app_data=None) -> None:
        """Open the render modal dialog.

        Checks if a render is already in progress before opening.
        """
        if dpg is None:
            return
        if self.app.render_orchestrator.render_future is not None and not self.app.render_orchestrator.render_future.done():
            dpg.set_value("status_text", "Render already in progress...")
            return

        # Rebuild modal if PDE changed (to update BC options)
        if self.modal_id is not None:
            dpg.delete_item(self.modal_id)
            self.modal_id = None
            self.bc_field_ids.clear()
            self.bc_enum_choices.clear()
            self.bc_field_defs.clear()
            
        self.ensure_modal()
        self.update_values()

        if self.modal_id is not None:
            dpg.configure_item(self.modal_id, show=True)
            self.modal_open = True

    def close(self) -> None:
        """Close the render modal dialog."""
        if dpg is None or self.modal_id is None:
            return
        dpg.configure_item(self.modal_id, show=False)
        self.modal_open = False

    def update_values(self) -> None:
        """Update modal form values from current app state."""
        if dpg is None:
            return

        with self.app.state_lock:
            settings = replace(self.app.state.render_settings)
            streamlength = self.app.state.project.streamlength_factor
            project = self.app.state.project

        if self.supersample_radio_id is not None:
            dpg.set_value(self.supersample_radio_id, _label_for_supersample(settings.supersample))

        if self.multiplier_radio_id is not None:
            dpg.set_value(self.multiplier_radio_id, _label_for_multiplier(settings.multiplier))

        if self.passes_input_id is not None:
            dpg.set_value(self.passes_input_id, int(settings.num_passes))

        if self.streamlength_input_id is not None:
            dpg.set_value(self.streamlength_input_id, float(streamlength))

        if self.margin_input_id is not None:
            dpg.set_value(self.margin_input_id, float(settings.margin))

        if self.solve_scale_slider_id is not None:
            dpg.set_value(self.solve_scale_slider_id, float(settings.solve_scale))

        if self.seed_input_id is not None:
            dpg.set_value(self.seed_input_id, int(settings.noise_seed))

        if self.sigma_input_id is not None:
            dpg.set_value(self.sigma_input_id, float(settings.noise_sigma))

        if self.use_mask_checkbox_id is not None:
            dpg.set_value(self.use_mask_checkbox_id, bool(settings.use_mask))

        if self.edge_gain_strength_slider_id is not None:
            dpg.set_value(self.edge_gain_strength_slider_id, float(settings.edge_gain_strength))

        if self.edge_gain_power_slider_id is not None:
            dpg.set_value(self.edge_gain_power_slider_id, float(settings.edge_gain_power))

        # Update boundary condition controls
        from flowcol.pde import PDERegistry
        pde = PDERegistry.get_active()
        bc_map = resolve_bc_map(project, pde)

        if self.bc_field_ids:
            for edge, field_map in self.bc_field_ids.items():
                for field_name, widget_id in field_map.items():
                    field = self.bc_field_defs.get(field_name)
                    if field is None or not dpg.does_item_exist(widget_id):
                        continue
                    val = bc_map.get(edge, {}).get(field_name, field.default)
                    if field.field_type == "enum":
                        choices = self.bc_enum_choices.get(edge, {}).get(field_name, [])
                        label = None
                        for lbl, v in choices:
                            if v == val:
                                label = lbl
                                break
                        if label is None and choices:
                            label = choices[0][0]
                        elif label is None:
                            label = str(val)
                        dpg.set_value(widget_id, label)
                    elif field.field_type == "bool":
                        dpg.set_value(widget_id, bool(val))
                    elif field.field_type == "int":
                        dpg.set_value(widget_id, int(val))
                    else:
                        dpg.set_value(widget_id, float(val))

        # Update field visibility after values are set (e.g., show/hide based on BC type)
        self._update_bc_field_visibility()

    def on_cancel(self, sender=None, app_data=None) -> None:
        """Handle cancel button click - closes modal without changes."""
        self.close()

    def on_apply(self, sender=None, app_data=None) -> None:
        """Handle apply/render button click - updates settings and starts render job."""
        if dpg is None:
            return

        # Read form values
        supersample_label = dpg.get_value(self.supersample_radio_id) if self.supersample_radio_id else SUPERSAMPLE_LABELS[0]
        multiplier_label = dpg.get_value(self.multiplier_radio_id) if self.multiplier_radio_id else RESOLUTION_LABELS[0]

        supersample = SUPERSAMPLE_LOOKUP.get(supersample_label, SUPERSAMPLE_CHOICES[0])
        multiplier = RESOLUTION_LOOKUP.get(multiplier_label, RESOLUTION_CHOICES[0])

        passes = int(dpg.get_value(self.passes_input_id)) if self.passes_input_id is not None else defaults.DEFAULT_RENDER_PASSES
        streamlength = float(dpg.get_value(self.streamlength_input_id)) if self.streamlength_input_id is not None else defaults.DEFAULT_STREAMLENGTH_FACTOR
        margin = float(dpg.get_value(self.margin_input_id)) if self.margin_input_id is not None else defaults.DEFAULT_PADDING_MARGIN
        solve_scale = float(dpg.get_value(self.solve_scale_slider_id)) if self.solve_scale_slider_id is not None else defaults.DEFAULT_SOLVE_SCALE
        noise_seed = int(dpg.get_value(self.seed_input_id)) if self.seed_input_id is not None else defaults.DEFAULT_NOISE_SEED
        noise_sigma = float(dpg.get_value(self.sigma_input_id)) if self.sigma_input_id is not None else defaults.DEFAULT_NOISE_SIGMA
        use_mask = bool(dpg.get_value(self.use_mask_checkbox_id)) if self.use_mask_checkbox_id is not None else defaults.DEFAULT_USE_MASK
        edge_gain_strength = float(dpg.get_value(self.edge_gain_strength_slider_id)) if self.edge_gain_strength_slider_id is not None else defaults.DEFAULT_EDGE_GAIN_STRENGTH
        edge_gain_power = float(dpg.get_value(self.edge_gain_power_slider_id)) if self.edge_gain_power_slider_id is not None else defaults.DEFAULT_EDGE_GAIN_POWER

        # Read boundary condition controls
        from flowcol.pde import PDERegistry
        pde = PDERegistry.get_active()
        bc_map = resolve_bc_map(self.app.state.project, pde)

        if self.bc_field_ids:
            for edge, field_map in self.bc_field_ids.items():
                for field_name, widget_id in field_map.items():
                    field = self.bc_field_defs.get(field_name)
                    if field is None or not dpg.does_item_exist(widget_id):
                        continue
                    if field.field_type == "enum":
                        label = dpg.get_value(widget_id)
                        choices = self.bc_enum_choices.get(edge, {}).get(field_name, [])
                        value = None
                        for lbl, val in choices:
                            if lbl == label:
                                value = val
                                break
                        if value is None and choices:
                            value = choices[0][1]
                        bc_map.setdefault(edge, {})[field_name] = value
                    elif field.field_type == "bool":
                        bc_map.setdefault(edge, {})[field_name] = bool(dpg.get_value(widget_id))
                    elif field.field_type == "int":
                        bc_map.setdefault(edge, {})[field_name] = int(dpg.get_value(widget_id))
                    else:
                        bc_map.setdefault(edge, {})[field_name] = float(dpg.get_value(widget_id))

        legacy_bc = bc_map_to_legacy(bc_map)

        # Clamp to valid ranges
        passes = max(passes, 1)
        streamlength = max(streamlength, 1e-6)
        margin = max(margin, 0.0)
        solve_scale = max(defaults.MIN_SOLVE_SCALE, min(defaults.MAX_SOLVE_SCALE, solve_scale))
        noise_sigma = max(noise_sigma, 0.0)
        edge_gain_strength = max(defaults.MIN_EDGE_GAIN_STRENGTH, min(defaults.MAX_EDGE_GAIN_STRENGTH, edge_gain_strength))
        edge_gain_power = max(defaults.MIN_EDGE_GAIN_POWER, min(defaults.MAX_EDGE_GAIN_POWER, edge_gain_power))

        # Update app state
        with self.app.state_lock:
            actions.set_supersample(self.app.state, supersample)
            actions.set_render_multiplier(self.app.state, multiplier)
            actions.set_num_passes(self.app.state, passes)
            actions.set_margin(self.app.state, margin)
            actions.set_noise_seed(self.app.state, noise_seed)
            actions.set_noise_sigma(self.app.state, noise_sigma)
            actions.set_streamlength_factor(self.app.state, streamlength)
            actions.set_solve_scale(self.app.state, solve_scale)
            self.app.state.render_settings.use_mask = use_mask
            self.app.state.render_settings.edge_gain_strength = edge_gain_strength
            self.app.state.render_settings.edge_gain_power = edge_gain_power
            # Update boundary conditions
            self.app.state.project.pde_bc[pde.name] = bc_map
            self.app.state.project.boundary_top = legacy_bc.get("top", self.app.state.project.boundary_top)
            self.app.state.project.boundary_bottom = legacy_bc.get("bottom", self.app.state.project.boundary_bottom)
            self.app.state.project.boundary_left = legacy_bc.get("left", self.app.state.project.boundary_left)
            self.app.state.project.boundary_right = legacy_bc.get("right", self.app.state.project.boundary_right)

        self.close()
        # Start the render job via orchestrator
        self.app.render_orchestrator.start_job()
