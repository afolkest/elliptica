"""Render modal dialog controller for Elliptica UI."""

from dataclasses import replace
from typing import Optional, TYPE_CHECKING

from elliptica import defaults
from elliptica.app import actions
from elliptica.pde.boundary_utils import resolve_bc_map, bc_map_to_legacy

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


# Constants for render settings
RESOLUTION_CHOICES = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
RESOLUTION_LABELS = ["1× (canvas)", "1.5×", "2×", "3×", "4×", "6×", "8×"]

class RenderModalController:
    """Controller for render settings modal dialog."""

    def __init__(self, app: "EllipticaApp"):
        """Initialize controller with reference to main app.

        Args:
            app: The main EllipticaApp instance
        """
        self.app = app

        # Modal window and state
        self.modal_id: Optional[int] = None
        self.modal_open: bool = False

        # Widget IDs for form controls
        self.multiplier_input_id: Optional[int] = None
        self.resolution_preview_id: Optional[int] = None
        self.passes_input_id: Optional[int] = None
        self.streamlength_input_id: Optional[int] = None
        self.margin_input_id: Optional[int] = None
        self.seed_input_id: Optional[int] = None
        self.sigma_input_id: Optional[int] = None
        self.solve_scale_slider_id: Optional[int] = None
        self.edge_gain_strength_slider_id: Optional[int] = None
        self.edge_gain_power_slider_id: Optional[int] = None
        self.domain_edge_gain_strength_slider_id: Optional[int] = None
        self.domain_edge_gain_power_slider_id: Optional[int] = None

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
            height=750,
        ) as modal:
            self.modal_id = modal

            # Resolution multiplier with +/- buttons
            with dpg.group(horizontal=True):
                resolution_label = dpg.add_text("Resolution")
                with dpg.tooltip(resolution_label):
                    dpg.add_text("Output size relative to canvas. Higher = more detail but slower. 1× matches canvas exactly.")
            with dpg.group(horizontal=True):
                dpg.add_button(label="-", width=30, callback=self._on_resolution_decrease, tag="res_decrease_btn")
                self.multiplier_input_id = dpg.add_input_float(
                    default_value=1.0,
                    min_value=0.5,
                    max_value=8.0,
                    min_clamped=True,
                    max_clamped=True,
                    step=0,
                    format="%.1f×",
                    width=80,
                    callback=self._on_resolution_input,
                    tag="resolution_input",
                )
                dpg.add_button(label="+", width=30, callback=self._on_resolution_increase, tag="res_increase_btn")
                dpg.add_spacer(width=10)
                self.resolution_preview_id = dpg.add_text("", color=(120, 120, 120), tag="resolution_preview")

            dpg.add_spacer(height=12)

            with dpg.group(horizontal=True):
                self.streamlength_input_id = dpg.add_slider_float(
                    label="",
                    default_value=defaults.DEFAULT_STREAMLENGTH_FACTOR * 1024.0,
                    min_value=10.0,
                    max_value=150.0,
                    format="%.0f",
                    width=250,
                )
                streamlen_label = dpg.add_text("Streamline Length")
                with dpg.tooltip(streamlen_label):
                    dpg.add_text("How far each streamline extends. Longer = smoother flow appearance, shorter = more detailed/grainy.")

            with dpg.group(horizontal=True):
                self.margin_input_id = dpg.add_slider_float(
                    label="",
                    default_value=defaults.DEFAULT_PADDING_MARGIN,
                    min_value=0.0,
                    max_value=0.2,
                    format="%.2f",
                    width=250,
                )
                margin_label = dpg.add_text("Padding Margin")
                with dpg.tooltip(margin_label):
                    dpg.add_text("Extra space around the canvas for boundary conditions. Prevents edge artifacts.")

            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                self.edge_gain_strength_slider_id = dpg.add_slider_float(
                    label="",
                    default_value=defaults.DEFAULT_EDGE_GAIN_STRENGTH,
                    min_value=defaults.MIN_EDGE_GAIN_STRENGTH,
                    max_value=defaults.MAX_EDGE_GAIN_STRENGTH,
                    format="%.2f",
                    width=250,
                )
                edge_strength_label = dpg.add_text("Edge Halo Strength")
                with dpg.tooltip(edge_strength_label):
                    dpg.add_text("Brightness boost near boundary edges. Creates a glowing halo effect around objects.")

            with dpg.group(horizontal=True):
                self.edge_gain_power_slider_id = dpg.add_slider_float(
                    label="",
                    default_value=defaults.DEFAULT_EDGE_GAIN_POWER,
                    min_value=defaults.MIN_EDGE_GAIN_POWER,
                    max_value=defaults.MAX_EDGE_GAIN_POWER,
                    format="%.2f",
                    width=250,
                )
                edge_power_label = dpg.add_text("Edge Halo Power")
                with dpg.tooltip(edge_power_label):
                    dpg.add_text("Controls how sharply the halo falls off from the edge. Higher = tighter glow, lower = broader spread.")

            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                self.domain_edge_gain_strength_slider_id = dpg.add_slider_float(
                    label="",
                    default_value=defaults.DEFAULT_DOMAIN_EDGE_GAIN_STRENGTH,
                    min_value=defaults.MIN_DOMAIN_EDGE_GAIN_STRENGTH,
                    max_value=defaults.MAX_DOMAIN_EDGE_GAIN_STRENGTH,
                    format="%.2f",
                    width=250,
                )
                domain_strength_label = dpg.add_text("Border Halo Strength")
                with dpg.tooltip(domain_strength_label):
                    dpg.add_text("Brightness boost at canvas borders (domain edges). Creates a glowing frame effect.")

            with dpg.group(horizontal=True):
                self.domain_edge_gain_power_slider_id = dpg.add_slider_float(
                    label="",
                    default_value=defaults.DEFAULT_DOMAIN_EDGE_GAIN_POWER,
                    min_value=defaults.MIN_DOMAIN_EDGE_GAIN_POWER,
                    max_value=defaults.MAX_DOMAIN_EDGE_GAIN_POWER,
                    format="%.2f",
                    width=250,
                )
                domain_power_label = dpg.add_text("Border Halo Power")
                with dpg.tooltip(domain_power_label):
                    dpg.add_text("Controls how sharply the border halo falls off. Higher = tighter glow, lower = broader spread.")

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)

            # Advanced settings (collapsible)
            with dpg.collapsing_header(label="Advanced", default_open=False):
                with dpg.group(horizontal=True):
                    self.solve_scale_slider_id = dpg.add_slider_float(
                        label="",
                        default_value=defaults.DEFAULT_SOLVE_SCALE,
                        min_value=defaults.MIN_SOLVE_SCALE,
                        max_value=defaults.MAX_SOLVE_SCALE,
                        format="%.2f",
                        width=250,
                    )
                    solve_label = dpg.add_text("Solve Scale")
                    with dpg.tooltip(solve_label):
                        dpg.add_text("Resolution for solving the PDE. Lower = faster but less precise.")

                with dpg.group(horizontal=True):
                    self.passes_input_id = dpg.add_slider_int(
                        label="",
                        default_value=defaults.DEFAULT_RENDER_PASSES,
                        min_value=1,
                        max_value=5,
                        width=250,
                    )
                    passes_label = dpg.add_text("LIC Passes")
                    with dpg.tooltip(passes_label):
                        dpg.add_text("Line integral convolution iterations. More = smoother streamlines, but slower.")

                with dpg.group(horizontal=True):
                    self.sigma_input_id = dpg.add_slider_float(
                        label="",
                        default_value=defaults.DEFAULT_NOISE_SIGMA,
                        min_value=0.0,
                        max_value=2.5,
                        format="%.2f",
                        width=250,
                    )
                    blur_label = dpg.add_text("Noise Blur")
                    with dpg.tooltip(blur_label):
                        dpg.add_text("Blur applied to noise texture. Higher = softer, more painterly look.")

                with dpg.group(horizontal=True):
                    self.seed_input_id = dpg.add_input_int(
                        label="",
                        step=1,
                        min_clamped=False,
                        width=160,
                    )
                    seed_label = dpg.add_text("Noise Seed")
                    with dpg.tooltip(seed_label):
                        dpg.add_text("Random seed for noise. Same seed = reproducible results.")

            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            # Dynamic Boundary Conditions Section (per-PDE)
            from elliptica.pde import PDERegistry
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

        # Consistent width for all controls (fits in 420px modal with padding)
        CONTROL_WIDTH = 180
        INDENT = 15

        edges = ["top", "right", "bottom", "left"]
        for edge in edges:
            self.bc_field_ids[edge] = {}
            self.bc_enum_choices[edge] = {}
            with dpg.collapsing_header(label=edge.title(), default_open=True, parent=parent):
                for field in bc_fields:
                    is_bc_type = field.name == "bc_type"

                    # Create horizontal group with indent (store group_id for visibility)
                    with dpg.group(horizontal=True) as group_id:
                        dpg.add_spacer(width=INDENT)

                        if field.field_type == "enum":
                            # Simplified labels for bc_type
                            if is_bc_type:
                                short_labels = [lbl.split(" (")[0] if " (" in lbl else lbl for lbl, _ in field.choices]
                                items = short_labels
                            else:
                                items = [lbl for lbl, _ in field.choices]
                            self.bc_enum_choices[edge][field.name] = list(field.choices)
                            widget_id = dpg.add_combo(
                                label="" if is_bc_type else field.display_name,
                                items=items,
                                width=CONTROL_WIDTH,
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
                                width=CONTROL_WIDTH,
                                min_value=field.min_value if field.min_value is not None else 0,
                                max_value=field.max_value if field.max_value is not None else 2147483647,
                            )
                        else:
                            widget_id = dpg.add_slider_float(
                                label=field.display_name,
                                width=CONTROL_WIDTH,
                                min_value=field.min_value if field.min_value is not None else 0.0,
                                max_value=field.max_value if field.max_value is not None else 1.0,
                                format="%.3f",
                            )

                        if getattr(field, "description", ""):
                            with dpg.tooltip(widget_id):
                                dpg.add_text(field.description)

                    # Store tuple of (group_id, widget_id) for visibility and value access
                    self.bc_field_ids[edge][field.name] = (group_id, widget_id)

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
            for field_name, ids in field_ids.items():
                group_id, widget_id = ids
                field_def = self.bc_field_defs.get(field_name)
                if field_def is None or not dpg.does_item_exist(widget_id):
                    continue
                if field_def.field_type == "enum":
                    # Convert label back to value (handle short labels)
                    label = dpg.get_value(widget_id)
                    choices = self.bc_enum_choices.get(edge, {}).get(field_name, [])
                    value = None
                    for lbl, v in choices:
                        # Match against full label or short label
                        short_lbl = lbl.split(" (")[0] if " (" in lbl else lbl
                        if lbl == label or short_lbl == label:
                            value = v
                            break
                    current_values[field_name] = value
                else:
                    current_values[field_name] = dpg.get_value(widget_id)

            # Now apply visibility rules to each field (show/hide the group)
            for field_name, ids in field_ids.items():
                group_id, widget_id = ids
                field_def = self.bc_field_defs.get(field_name)
                if field_def is None or not dpg.does_item_exist(group_id):
                    continue

                visible_when = getattr(field_def, "visible_when", None)
                if visible_when is None:
                    # No rule means always visible
                    dpg.configure_item(group_id, show=True)
                else:
                    # Check all conditions in visible_when dict
                    should_show = True
                    for dep_field, required_value in visible_when.items():
                        if current_values.get(dep_field) != required_value:
                            should_show = False
                            break
                    dpg.configure_item(group_id, show=should_show)

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

        if self.multiplier_input_id is not None:
            dpg.set_value(self.multiplier_input_id, float(settings.multiplier))
            self._update_resolution_preview()

        if self.passes_input_id is not None:
            dpg.set_value(self.passes_input_id, int(settings.num_passes))

        if self.streamlength_input_id is not None:
            # Display as "per 1024px" value (user-friendly)
            dpg.set_value(self.streamlength_input_id, float(streamlength * 1024.0))

        if self.margin_input_id is not None:
            dpg.set_value(self.margin_input_id, float(settings.margin))

        if self.solve_scale_slider_id is not None:
            dpg.set_value(self.solve_scale_slider_id, float(settings.solve_scale))

        if self.seed_input_id is not None:
            dpg.set_value(self.seed_input_id, int(settings.noise_seed))

        if self.sigma_input_id is not None:
            dpg.set_value(self.sigma_input_id, float(settings.noise_sigma))

        if self.edge_gain_strength_slider_id is not None:
            dpg.set_value(self.edge_gain_strength_slider_id, float(settings.edge_gain_strength))

        if self.edge_gain_power_slider_id is not None:
            dpg.set_value(self.edge_gain_power_slider_id, float(settings.edge_gain_power))

        if self.domain_edge_gain_strength_slider_id is not None:
            dpg.set_value(self.domain_edge_gain_strength_slider_id, float(settings.domain_edge_gain_strength))

        if self.domain_edge_gain_power_slider_id is not None:
            dpg.set_value(self.domain_edge_gain_power_slider_id, float(settings.domain_edge_gain_power))

        # Update boundary condition controls
        from elliptica.pde import PDERegistry
        pde = PDERegistry.get_active()
        bc_map = resolve_bc_map(project, pde)

        if self.bc_field_ids:
            for edge, field_map in self.bc_field_ids.items():
                for field_name, ids in field_map.items():
                    group_id, widget_id = ids
                    field = self.bc_field_defs.get(field_name)
                    if field is None or not dpg.does_item_exist(widget_id):
                        continue
                    val = bc_map.get(edge, {}).get(field_name, field.default)
                    if field.field_type == "enum":
                        choices = self.bc_enum_choices.get(edge, {}).get(field_name, [])
                        label = None
                        for lbl, v in choices:
                            if v == val:
                                # Use short label for bc_type
                                if field_name == "bc_type":
                                    label = lbl.split(" (")[0] if " (" in lbl else lbl
                                else:
                                    label = lbl
                                break
                        if label is None and choices:
                            lbl = choices[0][0]
                            label = lbl.split(" (")[0] if " (" in lbl and field_name == "bc_type" else lbl
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
        multiplier = float(dpg.get_value(self.multiplier_input_id)) if self.multiplier_input_id is not None else 1.0
        multiplier = max(0.5, min(8.0, multiplier))  # Clamp to valid range

        passes = int(dpg.get_value(self.passes_input_id)) if self.passes_input_id is not None else defaults.DEFAULT_RENDER_PASSES
        # Convert from "per 1024px" display value to internal factor
        streamlength_display = float(dpg.get_value(self.streamlength_input_id)) if self.streamlength_input_id is not None else (defaults.DEFAULT_STREAMLENGTH_FACTOR * 1024.0)
        streamlength = streamlength_display / 1024.0
        margin = float(dpg.get_value(self.margin_input_id)) if self.margin_input_id is not None else defaults.DEFAULT_PADDING_MARGIN
        solve_scale = float(dpg.get_value(self.solve_scale_slider_id)) if self.solve_scale_slider_id is not None else defaults.DEFAULT_SOLVE_SCALE
        noise_seed = int(dpg.get_value(self.seed_input_id)) if self.seed_input_id is not None else defaults.DEFAULT_NOISE_SEED
        noise_sigma = float(dpg.get_value(self.sigma_input_id)) if self.sigma_input_id is not None else defaults.DEFAULT_NOISE_SIGMA
        use_mask = True  # Always block streamlines at boundaries
        edge_gain_strength = float(dpg.get_value(self.edge_gain_strength_slider_id)) if self.edge_gain_strength_slider_id is not None else defaults.DEFAULT_EDGE_GAIN_STRENGTH
        edge_gain_power = float(dpg.get_value(self.edge_gain_power_slider_id)) if self.edge_gain_power_slider_id is not None else defaults.DEFAULT_EDGE_GAIN_POWER
        domain_edge_gain_strength = float(dpg.get_value(self.domain_edge_gain_strength_slider_id)) if self.domain_edge_gain_strength_slider_id is not None else defaults.DEFAULT_DOMAIN_EDGE_GAIN_STRENGTH
        domain_edge_gain_power = float(dpg.get_value(self.domain_edge_gain_power_slider_id)) if self.domain_edge_gain_power_slider_id is not None else defaults.DEFAULT_DOMAIN_EDGE_GAIN_POWER

        # Read boundary condition controls
        from elliptica.pde import PDERegistry
        pde = PDERegistry.get_active()
        bc_map = resolve_bc_map(self.app.state.project, pde)

        if self.bc_field_ids:
            for edge, field_map in self.bc_field_ids.items():
                for field_name, ids in field_map.items():
                    group_id, widget_id = ids
                    field = self.bc_field_defs.get(field_name)
                    if field is None or not dpg.does_item_exist(widget_id):
                        continue
                    if field.field_type == "enum":
                        label = dpg.get_value(widget_id)
                        choices = self.bc_enum_choices.get(edge, {}).get(field_name, [])
                        value = None
                        for lbl, val in choices:
                            # Match against full label or short label
                            short_lbl = lbl.split(" (")[0] if " (" in lbl else lbl
                            if lbl == label or short_lbl == label:
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
        domain_edge_gain_strength = max(defaults.MIN_DOMAIN_EDGE_GAIN_STRENGTH, min(defaults.MAX_DOMAIN_EDGE_GAIN_STRENGTH, domain_edge_gain_strength))
        domain_edge_gain_power = max(defaults.MIN_DOMAIN_EDGE_GAIN_POWER, min(defaults.MAX_DOMAIN_EDGE_GAIN_POWER, domain_edge_gain_power))

        # Update app state
        with self.app.state_lock:
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
            self.app.state.render_settings.domain_edge_gain_strength = domain_edge_gain_strength
            self.app.state.render_settings.domain_edge_gain_power = domain_edge_gain_power
            # Update boundary conditions
            self.app.state.project.pde_bc[pde.name] = bc_map
            self.app.state.project.boundary_top = legacy_bc.get("top", self.app.state.project.boundary_top)
            self.app.state.project.boundary_bottom = legacy_bc.get("bottom", self.app.state.project.boundary_bottom)
            self.app.state.project.boundary_left = legacy_bc.get("left", self.app.state.project.boundary_left)
            self.app.state.project.boundary_right = legacy_bc.get("right", self.app.state.project.boundary_right)

        self.close()
        # Start the render job via orchestrator
        self.app.render_orchestrator.start_job()

    # Resolution +/- button callbacks
    def _update_resolution_preview(self) -> None:
        """Update the resolution preview text with final pixel dimensions."""
        if dpg is None or self.resolution_preview_id is None or self.multiplier_input_id is None:
            return
        multiplier = float(dpg.get_value(self.multiplier_input_id))
        with self.app.state_lock:
            canvas_w, canvas_h = self.app.state.project.canvas_resolution
        final_w = int(canvas_w * multiplier)
        final_h = int(canvas_h * multiplier)
        dpg.set_value(self.resolution_preview_id, f"{final_w}×{final_h}")

    def _on_resolution_decrease(self, sender=None, app_data=None) -> None:
        """Decrease resolution multiplier by 0.5."""
        if dpg is None or self.multiplier_input_id is None:
            return
        current = float(dpg.get_value(self.multiplier_input_id))
        new_val = max(0.5, current - 0.5)
        dpg.set_value(self.multiplier_input_id, new_val)
        self._update_resolution_preview()

    def _on_resolution_increase(self, sender=None, app_data=None) -> None:
        """Increase resolution multiplier by 0.5."""
        if dpg is None or self.multiplier_input_id is None:
            return
        current = float(dpg.get_value(self.multiplier_input_id))
        new_val = min(8.0, current + 0.5)
        dpg.set_value(self.multiplier_input_id, new_val)
        self._update_resolution_preview()

    def _on_resolution_input(self, sender=None, app_data=None) -> None:
        """Handle direct input to resolution field."""
        self._update_resolution_preview()
