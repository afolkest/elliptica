"""Render modal dialog controller for FlowCol UI."""

from dataclasses import replace
from typing import Optional, TYPE_CHECKING

from flowcol import defaults
from flowcol.app import actions

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
        self.poisson_scale_slider_id: Optional[int] = None
        self.use_mask_checkbox_id: Optional[int] = None
        self.edge_gain_strength_slider_id: Optional[int] = None
        self.edge_gain_power_slider_id: Optional[int] = None

        # Generic Boundary Condition Controls
        # Map: edge_name -> combo_id
        self.bc_combo_ids: dict[str, int] = {}

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
            self.poisson_scale_slider_id = dpg.add_slider_float(
                label="Poisson Preview Scale",
                default_value=defaults.DEFAULT_POISSON_SCALE,
                min_value=defaults.MIN_POISSON_SCALE,
                max_value=defaults.MAX_POISSON_SCALE,
                format="%.2f",
                width=250,
            )
            dpg.add_text("Set <1.0 for faster preview solves (bilinear upsample).")

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
            
            # Dynamic Boundary Conditions Section
            from flowcol.pde import PDERegistry
            pde = PDERegistry.get_active()
            
            if pde.global_bc_options:
                dpg.add_text("Boundary Conditions")
                dpg.add_spacer(height=5)
                
                bc_labels = list(pde.global_bc_options.keys())
                
                # Cross-shaped layout for boundary controls
                with dpg.table(header_row=False, borders_innerH=False, borders_innerV=False,
                              borders_outerH=False, borders_outerV=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=100)
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=120)
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=100)

                    # Row 0: Top boundary centered
                    with dpg.table_row():
                        dpg.add_text("")
                        self.bc_combo_ids["top"] = dpg.add_combo(items=bc_labels, width=100)
                        dpg.add_text("")

                    # Row 1: Left and Right boundaries
                    with dpg.table_row():
                        self.bc_combo_ids["left"] = dpg.add_combo(items=bc_labels, width=100)
                        dpg.add_text("   (Domain)   ")
                        self.bc_combo_ids["right"] = dpg.add_combo(items=bc_labels, width=100)

                    # Row 2: Bottom boundary centered
                    with dpg.table_row():
                        dpg.add_text("")
                        self.bc_combo_ids["bottom"] = dpg.add_combo(items=bc_labels, width=100)
                        dpg.add_text("")
            else:
                dpg.add_text("No global boundary conditions available.")

            dpg.add_spacer(height=20)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Render", width=140, callback=self.on_apply)
                dpg.add_button(label="Cancel", width=140, callback=self.on_cancel)

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
            self.bc_combo_ids.clear()
            
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

        if self.poisson_scale_slider_id is not None:
            dpg.set_value(self.poisson_scale_slider_id, float(settings.poisson_scale))

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

        # Update boundary condition combos
        from flowcol.pde import PDERegistry
        pde = PDERegistry.get_active()
        
        # Reverse lookup: value -> label
        bc_lookup = {v: k for k, v in pde.global_bc_options.items()}
        
        if "top" in self.bc_combo_ids:
            label = bc_lookup.get(project.boundary_top, list(pde.global_bc_options.keys())[0])
            dpg.set_value(self.bc_combo_ids["top"], label)
            
        if "bottom" in self.bc_combo_ids:
            label = bc_lookup.get(project.boundary_bottom, list(pde.global_bc_options.keys())[0])
            dpg.set_value(self.bc_combo_ids["bottom"], label)
            
        if "left" in self.bc_combo_ids:
            label = bc_lookup.get(project.boundary_left, list(pde.global_bc_options.keys())[0])
            dpg.set_value(self.bc_combo_ids["left"], label)
            
        if "right" in self.bc_combo_ids:
            label = bc_lookup.get(project.boundary_right, list(pde.global_bc_options.keys())[0])
            dpg.set_value(self.bc_combo_ids["right"], label)

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
        poisson_scale = float(dpg.get_value(self.poisson_scale_slider_id)) if self.poisson_scale_slider_id is not None else defaults.DEFAULT_POISSON_SCALE
        noise_seed = int(dpg.get_value(self.seed_input_id)) if self.seed_input_id is not None else defaults.DEFAULT_NOISE_SEED
        noise_sigma = float(dpg.get_value(self.sigma_input_id)) if self.sigma_input_id is not None else defaults.DEFAULT_NOISE_SIGMA
        use_mask = bool(dpg.get_value(self.use_mask_checkbox_id)) if self.use_mask_checkbox_id is not None else defaults.DEFAULT_USE_MASK
        edge_gain_strength = float(dpg.get_value(self.edge_gain_strength_slider_id)) if self.edge_gain_strength_slider_id is not None else defaults.DEFAULT_EDGE_GAIN_STRENGTH
        edge_gain_power = float(dpg.get_value(self.edge_gain_power_slider_id)) if self.edge_gain_power_slider_id is not None else defaults.DEFAULT_EDGE_GAIN_POWER

        # Read boundary condition combos
        from flowcol.pde import PDERegistry
        pde = PDERegistry.get_active()
        
        boundary_top = 0
        boundary_bottom = 0
        boundary_left = 0
        boundary_right = 0
        
        if "top" in self.bc_combo_ids:
            label = dpg.get_value(self.bc_combo_ids["top"])
            boundary_top = pde.global_bc_options.get(label, 0)
            
        if "bottom" in self.bc_combo_ids:
            label = dpg.get_value(self.bc_combo_ids["bottom"])
            boundary_bottom = pde.global_bc_options.get(label, 0)
            
        if "left" in self.bc_combo_ids:
            label = dpg.get_value(self.bc_combo_ids["left"])
            boundary_left = pde.global_bc_options.get(label, 0)
            
        if "right" in self.bc_combo_ids:
            label = dpg.get_value(self.bc_combo_ids["right"])
            boundary_right = pde.global_bc_options.get(label, 0)

        # Clamp to valid ranges
        passes = max(passes, 1)
        streamlength = max(streamlength, 1e-6)
        margin = max(margin, 0.0)
        poisson_scale = max(defaults.MIN_POISSON_SCALE, min(defaults.MAX_POISSON_SCALE, poisson_scale))
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
            actions.set_poisson_scale(self.app.state, poisson_scale)
            self.app.state.render_settings.use_mask = use_mask
            self.app.state.render_settings.edge_gain_strength = edge_gain_strength
            self.app.state.render_settings.edge_gain_power = edge_gain_power
            # Update boundary conditions
            self.app.state.project.boundary_top = boundary_top
            self.app.state.project.boundary_bottom = boundary_bottom
            self.app.state.project.boundary_left = boundary_left
            self.app.state.project.boundary_right = boundary_right

        self.close()
        # Start the render job via orchestrator
        self.app.render_orchestrator.start_job()
