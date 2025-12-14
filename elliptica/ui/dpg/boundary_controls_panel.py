"""Boundary controls panel for Elliptica UI - generic parameter sliders."""

from typing import Optional, Dict, Any, TYPE_CHECKING
from elliptica.app import actions
from elliptica.pde import PDERegistry

if TYPE_CHECKING:
    from elliptica.ui.dpg.app import EllipticaApp

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None  # type: ignore


class BoundaryControlsPanel:
    """Controller for generic boundary object controls (dynamic sliders)."""

    def __init__(self, app: "EllipticaApp"):
        """Initialize controller with reference to main app."""
        self.app = app

        # Widget IDs
        self.container_id: Optional[int] = None
        # Map: boundary_idx -> header_id (for collapsing headers)
        self.header_ids: Dict[int, int] = {}
        # Map: boundary_idx -> param_name -> slider_id (for boundary_params)
        self.slider_ids: Dict[int, Dict[str, int]] = {}
        # Map: boundary_idx -> field_name -> widget_id (for boundary_fields)
        self.field_ids: Dict[int, Dict[str, int]] = {}
        # Cache for enum choices: field_name -> [(label, value)]
        self.enum_choices: Dict[str, list] = {}
        # Cache for field definitions by name
        self.field_defs: Dict[str, Any] = {}

    def build_container(self, parent) -> None:
        """Build controls container in edit controls panel."""
        if dpg is None:
            return

        dpg.add_text("Object Properties", parent=parent)
        self.container_id = dpg.add_child_window(
            autosize_x=True,
            height=-10,  # Fill remaining space in parent
            border=False,
            tag="boundary_controls_child",
            no_scroll_with_mouse=False,
            parent=parent,
        )

    def rebuild_controls(self) -> None:
        """Rebuild all sliders based on active PDE metadata."""
        if dpg is None or self.container_id is None:
            return

        # Save collapsed state before destroying widgets
        # dpg.get_value on collapsing_header returns True if open, False if collapsed
        collapsed_state: Dict[int, bool] = {}
        for idx, header_id in self.header_ids.items():
            if dpg.does_item_exist(header_id):
                collapsed_state[idx] = not dpg.get_value(header_id)

        dpg.delete_item(self.container_id, children_only=True)
        self.header_ids.clear()
        self.slider_ids.clear()
        self.field_ids.clear()
        self.enum_choices.clear()
        self.field_defs.clear()

        with self.app.state_lock:
            boundaries = list(self.app.state.project.boundary_objects)
            selected_indices = set(self.app.state.selected_indices)

        # Get active PDE definition to know what sliders to build
        pde = PDERegistry.get_active()
        params_meta = pde.boundary_params
        fields_meta = getattr(pde, 'boundary_fields', [])

        # Cache field definitions
        self.field_defs = {f.name: f for f in fields_meta}
        for f in fields_meta:
            if f.field_type == "enum":
                self.enum_choices[f.name] = list(f.choices)

        if not boundaries:
            dpg.add_text("No boundaries loaded.", parent=self.container_id)
            return

        for idx, boundary in enumerate(boundaries):
            label = f"Object {idx + 1}"
            if idx in selected_indices:
                label += " (selected)"

            # Add boundary type to label if bc_type field exists
            bc_type_field = self.field_defs.get("bc_type")
            if bc_type_field and bc_type_field.field_type == "enum":
                current_bc = boundary.params.get("bc_type", bc_type_field.default)
                for lbl, val in bc_type_field.choices:
                    if val == current_bc:
                        # Extract short name (e.g., "Dirichlet" from "Dirichlet (fixed V)")
                        short_name = lbl.split(" (")[0] if " (" in lbl else lbl
                        label += f" - {short_name}"
                        break

            header_id = dpg.add_collapsing_header(label=label, default_open=True, parent=self.container_id)
            self.header_ids[idx] = header_id
            with dpg.group(parent=header_id):
                self.slider_ids[idx] = {}
                self.field_ids[idx] = {}

                # Build rich boundary_fields (enums, floats with visibility rules, etc.)
                for field in fields_meta:
                    current_val = boundary.params.get(field.name, field.default)
                    widget_id = self._build_field_widget(idx, field, current_val)
                    if widget_id is not None:
                        self.field_ids[idx][field.name] = widget_id

                # Dynamic sliders for PDE parameters (legacy boundary_params)
                for param in params_meta:
                    current_val = boundary.params.get(param.name, param.default_value)

                    with dpg.group(horizontal=True):
                        slider_id = dpg.add_slider_float(
                            label="",
                            default_value=float(current_val),
                            min_value=param.min_value,
                            max_value=param.max_value,
                            format="%.3f",
                            callback=self.on_param_slider,
                            user_data={"idx": idx, "param": param.name},
                        )
                        param_label = dpg.add_text(param.display_name)
                        if param.description:
                            with dpg.tooltip(param_label):
                                dpg.add_text(param.description)
                    self.slider_ids[idx][param.name] = slider_id

                # Edge smoothing slider (inline, no header)
                dpg.add_slider_float(
                    label="Edge Smooth",
                    default_value=float(boundary.edge_smooth_sigma),
                    min_value=0.0,
                    max_value=5.0,
                    format="%.1f px",
                    callback=self.on_edge_smooth_slider,
                    user_data=idx,
                )

        # Set initial visibility based on field values
        self._update_field_visibility()

        # Restore collapsed state for headers that existed before
        for idx, header_id in self.header_ids.items():
            if idx in collapsed_state and collapsed_state[idx]:
                dpg.set_value(header_id, False)  # Collapse it

    def update_header_labels(self) -> None:
        """Update collapsing header labels without rebuilding widgets."""
        if dpg is None or not self.header_ids:
            return

        with self.app.state_lock:
            boundaries = list(self.app.state.project.boundary_objects)
            selected_indices = set(self.app.state.selected_indices)

        bc_type_field = self.field_defs.get("bc_type")

        for idx, header_id in self.header_ids.items():
            if idx >= len(boundaries) or not dpg.does_item_exist(header_id):
                continue

            boundary = boundaries[idx]
            label = f"Object {idx + 1}"
            if idx in selected_indices:
                label += " (selected)"

            # Add boundary type to label if bc_type field exists
            if bc_type_field and bc_type_field.field_type == "enum":
                current_bc = boundary.params.get("bc_type", bc_type_field.default)
                for lbl, val in bc_type_field.choices:
                    if val == current_bc:
                        short_name = lbl.split(" (")[0] if " (" in lbl else lbl
                        label += f" - {short_name}"
                        break

            dpg.configure_item(header_id, label=label)

    def _build_field_widget(self, idx: int, field, current_val) -> Optional[int]:
        """Build a widget for a boundary field. Returns group ID (for visibility control)."""
        if dpg is None:
            return None

        # Wrap in a horizontal group: widget first, then label with tooltip
        group_id = dpg.add_group(horizontal=True)

        if field.field_type == "enum":
            labels = [lbl for lbl, _ in field.choices]
            # Find current label
            current_label = labels[0] if labels else ""
            for lbl, val in field.choices:
                if val == current_val:
                    current_label = lbl
                    break
            dpg.add_combo(
                label="",
                items=labels,
                default_value=current_label,
                width=180,
                callback=self._on_field_changed,
                user_data={"idx": idx, "field": field.name},
                parent=group_id,
            )
        elif field.field_type == "bool":
            dpg.add_checkbox(
                label="",
                default_value=bool(current_val),
                callback=self._on_field_changed,
                user_data={"idx": idx, "field": field.name},
                parent=group_id,
            )
        elif field.field_type == "int":
            dpg.add_input_int(
                label="",
                default_value=int(current_val),
                width=120,
                min_value=int(field.min_value) if field.min_value is not None else 0,
                max_value=int(field.max_value) if field.max_value is not None else 2147483647,
                callback=self._on_field_changed,
                user_data={"idx": idx, "field": field.name},
                parent=group_id,
            )
        else:  # float
            dpg.add_slider_float(
                label="",
                default_value=float(current_val),
                min_value=field.min_value if field.min_value is not None else -1e9,
                max_value=field.max_value if field.max_value is not None else 1e9,
                format="%.3f",
                callback=self._on_field_changed,
                user_data={"idx": idx, "field": field.name},
                parent=group_id,
            )

        # Label with tooltip (after widget)
        field_label = dpg.add_text(field.display_name, parent=group_id)
        if field.description:
            with dpg.tooltip(field_label):
                dpg.add_text(field.description)

        return group_id

    def _on_field_changed(self, sender, app_data, user_data) -> None:
        """Handle boundary field change."""
        if dpg is None or user_data is None:
            return

        idx = int(user_data["idx"])
        field_name = user_data["field"]
        field_def = self.field_defs.get(field_name)
        if field_def is None:
            return

        # Convert value based on field type
        if field_def.field_type == "enum":
            # app_data is the selected label, convert to value
            value = None
            for lbl, val in self.enum_choices.get(field_name, []):
                if lbl == app_data:
                    value = val
                    break
            if value is None:
                return
        elif field_def.field_type == "bool":
            value = bool(app_data)
        elif field_def.field_type == "int":
            value = int(app_data)
        else:
            value = float(app_data)

        with self.app.state_lock:
            if idx < len(self.app.state.project.boundary_objects):
                self.app.state.project.boundary_objects[idx].params[field_name] = value

        self.app.canvas_renderer.mark_dirty()
        dpg.set_value("status_text", f"Obj {idx + 1} {field_name} = {value}")

        # Update visibility for conditional fields
        self._update_field_visibility()

        # Update header labels if bc_type changed
        if field_name == "bc_type":
            self.update_header_labels()

    def _get_widget_from_group(self, group_id: int) -> Optional[int]:
        """Get the actual widget (combo/slider/etc) from a field group. Returns None if not found."""
        if dpg is None or not dpg.does_item_exist(group_id):
            return None
        # Children are [widget, label_text] - we want the widget (first child)
        children = dpg.get_item_children(group_id, 1)  # slot 1 = most children
        if children and len(children) >= 1:
            return children[0]  # First child is the widget
        return None

    def _update_field_visibility(self) -> None:
        """Show/hide fields based on their visible_when rules."""
        if dpg is None:
            return

        for idx, field_map in self.field_ids.items():
            # Collect current values for this boundary
            current_values = {}
            for field_name, group_id in field_map.items():
                field_def = self.field_defs.get(field_name)
                if field_def is None or not dpg.does_item_exist(group_id):
                    continue
                widget_id = self._get_widget_from_group(group_id)
                if widget_id is None:
                    continue
                if field_def.field_type == "enum":
                    label = dpg.get_value(widget_id)
                    for lbl, val in self.enum_choices.get(field_name, []):
                        if lbl == label:
                            current_values[field_name] = val
                            break
                else:
                    current_values[field_name] = dpg.get_value(widget_id)

            # Apply visibility rules (show/hide the group, which contains label+widget)
            for field_name, group_id in field_map.items():
                field_def = self.field_defs.get(field_name)
                if field_def is None or not dpg.does_item_exist(group_id):
                    continue

                visible_when = getattr(field_def, 'visible_when', None)
                if visible_when is None:
                    dpg.configure_item(group_id, show=True)
                else:
                    should_show = all(
                        current_values.get(dep_field) == required_value
                        for dep_field, required_value in visible_when.items()
                    )
                    dpg.configure_item(group_id, show=should_show)

    def update_slider_labels(self, skip_idx: Optional[int] = None) -> None:
        """Update slider labels and values to reflect current state."""
        if dpg is None or not self.slider_ids:
            return

        with self.app.state_lock:
            boundaries = list(self.app.state.project.boundary_objects)

        for idx, param_map in list(self.slider_ids.items()):
            if idx >= len(boundaries):
                continue

            if skip_idx is not None and idx == skip_idx:
                continue

            boundary = boundaries[idx]
            for param_name, slider_id in param_map.items():
                if dpg.does_item_exist(slider_id):
                    val = boundary.params.get(param_name, 0.0)
                    dpg.set_value(slider_id, float(val))

    def on_param_slider(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle generic parameter slider change."""
        if dpg is None or user_data is None:
            return

        idx = int(user_data["idx"])
        param_name = user_data["param"]
        value = float(app_data)

        with self.app.state_lock:
            if idx < len(self.app.state.project.boundary_objects):
                # Update params dict directly
                # TODO: Add an action for this to support undo/redo
                self.app.state.project.boundary_objects[idx].params[param_name] = value

        self.app.canvas_renderer.mark_dirty()
        dpg.set_value("status_text", f"Obj {idx + 1} {param_name} = {value:.3f}")
        self.update_slider_labels(skip_idx=idx)

    def on_edge_smooth_slider(self, sender=None, app_data=None, user_data=None) -> None:
        """Handle edge smoothing slider change."""
        if dpg is None or user_data is None:
            return

        idx = int(user_data)
        value = float(app_data)

        with self.app.state_lock:
            if idx < len(self.app.state.project.boundary_objects):
                self.app.state.project.boundary_objects[idx].edge_smooth_sigma = value

        self.app.canvas_renderer.mark_dirty()
        dpg.set_value("status_text", f"Obj {idx + 1} edge smoothing = {value:.1f} px")
