#!/usr/bin/env python3
"""Standalone theme preview for Elliptica UI.

Run with: python tools/theme_preview.py

Steel Blue Soft base theme with options to tweak remaining settings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dearpygui.dearpygui as dpg

# Steel Blue Soft - the chosen base theme
BASE_COLORS = {
    # Window/frame backgrounds
    dpg.mvThemeCol_WindowBg: (32, 34, 45, 255),
    dpg.mvThemeCol_ChildBg: (38, 40, 50, 255),
    dpg.mvThemeCol_PopupBg: (44, 46, 56, 255),
    dpg.mvThemeCol_MenuBarBg: (32, 34, 45, 255),

    # Borders
    dpg.mvThemeCol_Border: (62, 64, 74, 80),
    dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

    # Frame backgrounds (inputs, sliders)
    dpg.mvThemeCol_FrameBg: (46, 48, 58, 255),
    dpg.mvThemeCol_FrameBgHovered: (54, 56, 66, 255),
    dpg.mvThemeCol_FrameBgActive: (62, 64, 74, 255),

    # Title bar
    dpg.mvThemeCol_TitleBg: (26, 28, 38, 255),
    dpg.mvThemeCol_TitleBgActive: (36, 38, 48, 255),
    dpg.mvThemeCol_TitleBgCollapsed: (26, 28, 38, 200),

    # Scrollbar
    dpg.mvThemeCol_ScrollbarBg: (32, 34, 45, 255),
    dpg.mvThemeCol_ScrollbarGrab: (66, 72, 82, 255),
    dpg.mvThemeCol_ScrollbarGrabHovered: (76, 82, 94, 255),
    dpg.mvThemeCol_ScrollbarGrabActive: (86, 92, 106, 255),

    # Buttons
    dpg.mvThemeCol_Button: (64, 70, 80, 255),
    dpg.mvThemeCol_ButtonHovered: (74, 80, 92, 255),
    dpg.mvThemeCol_ButtonActive: (84, 90, 104, 255),

    # Headers
    dpg.mvThemeCol_Header: (52, 54, 64, 255),
    dpg.mvThemeCol_HeaderHovered: (62, 64, 74, 255),
    dpg.mvThemeCol_HeaderActive: (72, 74, 84, 255),

    # Tabs
    dpg.mvThemeCol_Tab: (46, 48, 58, 255),
    dpg.mvThemeCol_TabHovered: (60, 62, 72, 255),
    dpg.mvThemeCol_TabActive: (54, 56, 66, 255),
    dpg.mvThemeCol_TabUnfocused: (42, 44, 54, 255),
    dpg.mvThemeCol_TabUnfocusedActive: (50, 52, 62, 255),

    # Slider grab
    dpg.mvThemeCol_SliderGrab: (100, 108, 122, 255),
    dpg.mvThemeCol_SliderGrabActive: (116, 124, 140, 255),

    # Checkmark
    dpg.mvThemeCol_CheckMark: (150, 160, 176, 255),

    # Text
    dpg.mvThemeCol_Text: (218, 222, 228, 255),
    dpg.mvThemeCol_TextDisabled: (134, 138, 144, 255),

    # Separator
    dpg.mvThemeCol_Separator: (62, 64, 74, 100),
    dpg.mvThemeCol_SeparatorHovered: (88, 90, 100, 255),
    dpg.mvThemeCol_SeparatorActive: (106, 108, 118, 255),

    # Resize grip
    dpg.mvThemeCol_ResizeGrip: (62, 64, 74, 45),
    dpg.mvThemeCol_ResizeGripHovered: (88, 90, 100, 180),
    dpg.mvThemeCol_ResizeGripActive: (106, 108, 118, 255),

    # Plot
    dpg.mvThemeCol_PlotHistogram: (112, 122, 140, 255),
    dpg.mvThemeCol_PlotHistogramHovered: (132, 142, 162, 255),
}

BASE_STYLES = {
    dpg.mvStyleVar_FrameRounding: 5,
    dpg.mvStyleVar_WindowRounding: 6,
    dpg.mvStyleVar_ChildRounding: 4,
    dpg.mvStyleVar_PopupRounding: 5,
    dpg.mvStyleVar_ScrollbarRounding: 6,
    dpg.mvStyleVar_GrabRounding: 3,
    dpg.mvStyleVar_TabRounding: 3,
    dpg.mvStyleVar_FramePadding: (8, 5),
    dpg.mvStyleVar_ItemSpacing: (8, 6),
    dpg.mvStyleVar_WindowPadding: (12, 12),
    dpg.mvStyleVar_FrameBorderSize: 0,
    dpg.mvStyleVar_WindowBorderSize: 0,
}

# Options to choose between
OPTIONS = {
    "rounding": {
        "label": "Corner Rounding",
        "choices": [
            ("1.0px", 1.0),
            ("1.5px", 1.5),
            ("2.0px", 2.0),
            ("2.5px", 2.5),
            ("3.0px", 3.0),
            ("3.5px", 3.5),
        ],
    },
    "selection_highlight": {
        "label": "Text Selection",
        "choices": [
            ("Steel subtle", (70, 80, 95, 120)),
            ("Steel medium", (80, 90, 110, 150)),
            ("Steel visible", (90, 100, 120, 180)),
            ("Blue tint", (60, 80, 120, 140)),
        ],
    },
    "modal_dim": {
        "label": "Modal Dimming",
        "choices": [
            ("Light", (0, 0, 0, 80)),
            ("Medium", (0, 0, 0, 120)),
            ("Dark", (0, 0, 0, 160)),
            ("Very dark", (0, 0, 0, 200)),
        ],
    },
}

# Current selections
current_selection = {
    "rounding": 3,  # Default to 2.5px (user's choice)
    "selection_highlight": 1,  # Default to medium
    "modal_dim": 1,  # Default to medium
}

theme_id = None


def _get_display_scale() -> float:
    if sys.platform != "darwin":
        return 1.0
    try:
        import subprocess
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=2
        )
        if "Retina" in result.stdout:
            return 2.0
    except Exception:
        pass
    return 1.0


def _find_dejavu_font() -> Path | None:
    try:
        import matplotlib
        mpl_data = Path(matplotlib.get_data_path())
        font_path = mpl_data / "fonts" / "ttf" / "DejaVuSans.ttf"
        if font_path.exists():
            return font_path
    except ImportError:
        pass
    return None


def setup_font(size: int = 14) -> int | None:
    font_path = _find_dejavu_font()
    if font_path is None:
        return None

    display_scale = _get_display_scale()
    raster_size = int(size * display_scale)

    with dpg.font_registry():
        with dpg.font(str(font_path), raster_size) as font_id:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            dpg.add_font_range(0x0370, 0x03FF)  # Greek
            dpg.add_font_range(0x2200, 0x22FF)  # Math operators
            dpg.add_font_range(0x2070, 0x209F)  # Sub/superscripts
            dpg.add_font_chars([0x2202, 0x2207, 0x03C6, 0x2099, 0x00B2, 0x00D7])

    dpg.bind_font(font_id)
    if display_scale > 1.0:
        dpg.set_global_font_scale(1.0 / display_scale)

    return font_id


def rebuild_theme():
    global theme_id

    if theme_id is not None and dpg.does_item_exist(theme_id):
        dpg.delete_item(theme_id)

    with dpg.theme() as theme_id:
        with dpg.theme_component(dpg.mvAll):
            # Base colors
            for color_id, rgba in BASE_COLORS.items():
                dpg.add_theme_color(color_id, rgba)

            # Base styles
            for style_id, value in BASE_STYLES.items():
                if isinstance(value, tuple):
                    dpg.add_theme_style(style_id, value[0], value[1])
                else:
                    dpg.add_theme_style(style_id, value)

            # Rounding override
            round_val = OPTIONS["rounding"]["choices"][current_selection["rounding"]][1]
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, round_val)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, round_val + 0.5)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, round_val)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, round_val)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, round_val)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, round_val)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, round_val)

            # Text selection highlight
            sel_val = OPTIONS["selection_highlight"]["choices"][current_selection["selection_highlight"]][1]
            dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, sel_val)

            # Modal dimming
            dim_val = OPTIONS["modal_dim"]["choices"][current_selection["modal_dim"]][1]
            dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg, dim_val)

    dpg.bind_theme(theme_id)


def on_option_change(sender, app_data, user_data):
    option_key, choice_idx = user_data
    current_selection[option_key] = choice_idx
    rebuild_theme()
    update_selection_display()


def update_selection_display():
    for key, opt in OPTIONS.items():
        idx = current_selection[key]
        label = opt["choices"][idx][0]
        dpg.set_value(f"{key}_display", f"Current: {label}")


def show_test_modal(sender, app_data):
    if not dpg.does_item_exist("test_modal"):
        with dpg.window(
            label="Test Modal",
            modal=True,
            tag="test_modal",
            width=300,
            height=150,
            no_resize=True,
        ):
            dpg.add_text("This is a modal dialog.")
            dpg.add_text("Check the background dimming.")
            dpg.add_spacer(height=20)
            dpg.add_button(label="Close", callback=lambda: dpg.configure_item("test_modal", show=False))

    # Center it
    vp_w = dpg.get_viewport_width()
    vp_h = dpg.get_viewport_height()
    dpg.configure_item("test_modal", pos=[(vp_w - 300) // 2, (vp_h - 150) // 2])
    dpg.configure_item("test_modal", show=True)


def build_option_selector(key: str, opt: dict, parent):
    """Build radio-button style selector for an option."""
    dpg.add_text(opt["label"], parent=parent)
    dpg.add_text("", tag=f"{key}_display", parent=parent, color=(150, 155, 165))

    with dpg.group(horizontal=True, parent=parent):
        for i, (label, _) in enumerate(opt["choices"]):
            dpg.add_button(
                label=label,
                callback=on_option_change,
                user_data=(key, i),
                width=100,
            )
    dpg.add_spacer(height=10, parent=parent)


def build_sample_widgets(parent):
    """Build sample widgets to preview theme."""
    with dpg.child_window(width=-1, height=100, border=True, parent=parent):
        dpg.add_text("Sample Panel")
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="Button")
            dpg.add_button(label="Another")
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Checkbox")
            dpg.add_checkbox(label="Checked", default_value=True)

    dpg.add_spacer(height=10, parent=parent)
    dpg.add_slider_float(label="Slider", default_value=0.5, width=200, parent=parent)
    dpg.add_input_text(label="Text Input", default_value="Select this text", width=200, parent=parent)
    dpg.add_combo(["Option A", "Option B", "Option C"], label="Combo", default_value="Option A", width=200, parent=parent)

    dpg.add_spacer(height=10, parent=parent)
    with dpg.collapsing_header(label="Collapsible Section", default_open=True, parent=parent):
        dpg.add_text("Nested content")
        dpg.add_button(label="Nested Button")

    dpg.add_spacer(height=10, parent=parent)
    dpg.add_text("Math: ∂φ/∂n × ∇²φ", parent=parent, color=(180, 185, 195))

    dpg.add_spacer(height=10, parent=parent)
    dpg.add_button(label="Test Modal Dialog", callback=show_test_modal, parent=parent)


def print_final_theme():
    """Print the current theme configuration for copying to elliptica."""
    print("\n" + "=" * 60)
    print("FINAL THEME CONFIGURATION - Steel Blue Soft")
    print("=" * 60)

    print("\nSelected options:")
    for key, opt in OPTIONS.items():
        idx = current_selection[key]
        label, value = opt["choices"][idx]
        print(f"  {opt['label']}: {label} = {value}")

    round_val = OPTIONS["rounding"]["choices"][current_selection["rounding"]][1]
    sel_val = OPTIONS["selection_highlight"]["choices"][current_selection["selection_highlight"]][1]
    dim_val = OPTIONS["modal_dim"]["choices"][current_selection["modal_dim"]][1]

    print("\nTo apply in elliptica/ui/dpg/theme.py:")
    print(f"  dpg.mvThemeCol_TextSelectedBg: {sel_val},")
    print(f"  dpg.mvThemeCol_ModalWindowDimBg: {dim_val},")
    print(f"\n  # Rounding: {round_val}px")
    print(f"  dpg.mvStyleVar_FrameRounding: {round_val},")
    print(f"  dpg.mvStyleVar_WindowRounding: {round_val + 0.5},")
    print(f"  dpg.mvStyleVar_ChildRounding: {round_val},")
    print(f"  dpg.mvStyleVar_PopupRounding: {round_val},")
    print(f"  dpg.mvStyleVar_ScrollbarRounding: {round_val},")
    print(f"  dpg.mvStyleVar_GrabRounding: {round_val},")
    print(f"  dpg.mvStyleVar_TabRounding: {round_val},")
    print("=" * 60)


def main():
    dpg.create_context()
    setup_font(14)

    dpg.create_viewport(title="Steel Blue Soft - Theme Options", width=500, height=700)

    with dpg.window(label="Theme Options", tag="primary_window"):
        dpg.add_text("Steel Blue Soft", color=(150, 160, 176))
        dpg.add_text("Choose remaining options:", color=(134, 138, 144))
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Option selectors
        for key, opt in OPTIONS.items():
            build_option_selector(key, opt, "primary_window")

        dpg.add_separator()
        dpg.add_spacer(height=10)

        dpg.add_text("Preview", color=(150, 160, 176))
        build_sample_widgets("primary_window")

        dpg.add_spacer(height=20)
        dpg.add_button(label="Print Final Config", callback=lambda: print_final_theme())

    rebuild_theme()
    update_selection_display()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
