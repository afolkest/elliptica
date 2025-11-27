#!/usr/bin/env python3
"""Standalone theme preview for Elliptica UI.

Run with: python tools/theme_preview.py

Shows sample widgets with different warm dark theme variations.
Toggle between themes and fonts using the buttons at the top.
"""

import sys
from pathlib import Path

# Add parent to path so we can import elliptica modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import dearpygui.dearpygui as dpg

# Font directory
FONTS_DIR = Path(__file__).parent.parent / "elliptica" / "assets" / "fonts"

# Available fonts
FONTS = {
    "DejaVu Sans": None,  # Will use matplotlib's bundled font
    "Source Sans 3": FONTS_DIR / "SourceSans3-Regular.ttf",
    "Fira Sans": FONTS_DIR / "FiraSans-Regular.ttf",
    "Nunito": FONTS_DIR / "Nunito-Regular.ttf",
}

# Theme definitions - warm dark "washi" aesthetics
THEMES = {
    "Charcoal Washi": {
        "description": "Warm charcoal with cream text, subtle earth accents",
        "colors": {
            # Window/frame backgrounds
            dpg.mvThemeCol_WindowBg: (35, 33, 32, 255),
            dpg.mvThemeCol_ChildBg: (40, 38, 36, 255),
            dpg.mvThemeCol_PopupBg: (45, 42, 40, 255),
            dpg.mvThemeCol_MenuBarBg: (35, 33, 32, 255),

            # Borders - very subtle
            dpg.mvThemeCol_Border: (60, 55, 50, 100),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            # Frame backgrounds (inputs, sliders)
            dpg.mvThemeCol_FrameBg: (50, 47, 44, 255),
            dpg.mvThemeCol_FrameBgHovered: (60, 56, 52, 255),
            dpg.mvThemeCol_FrameBgActive: (70, 65, 60, 255),

            # Title bar
            dpg.mvThemeCol_TitleBg: (30, 28, 27, 255),
            dpg.mvThemeCol_TitleBgActive: (40, 38, 36, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (30, 28, 27, 200),

            # Scrollbar
            dpg.mvThemeCol_ScrollbarBg: (35, 33, 32, 255),
            dpg.mvThemeCol_ScrollbarGrab: (70, 65, 58, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (85, 78, 70, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (100, 92, 82, 255),

            # Buttons - muted indigo accent
            dpg.mvThemeCol_Button: (65, 62, 75, 255),
            dpg.mvThemeCol_ButtonHovered: (80, 75, 92, 255),
            dpg.mvThemeCol_ButtonActive: (90, 85, 105, 255),

            # Headers (collapsing headers, tree nodes)
            dpg.mvThemeCol_Header: (55, 52, 48, 255),
            dpg.mvThemeCol_HeaderHovered: (65, 60, 55, 255),
            dpg.mvThemeCol_HeaderActive: (75, 70, 64, 255),

            # Tabs
            dpg.mvThemeCol_Tab: (45, 42, 40, 255),
            dpg.mvThemeCol_TabHovered: (65, 60, 55, 255),
            dpg.mvThemeCol_TabActive: (55, 52, 48, 255),
            dpg.mvThemeCol_TabUnfocused: (40, 38, 36, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (50, 47, 44, 255),

            # Slider grab
            dpg.mvThemeCol_SliderGrab: (120, 110, 95, 255),
            dpg.mvThemeCol_SliderGrabActive: (145, 135, 118, 255),

            # Checkmark, radio
            dpg.mvThemeCol_CheckMark: (180, 165, 140, 255),

            # Text
            dpg.mvThemeCol_Text: (240, 235, 225, 255),
            dpg.mvThemeCol_TextDisabled: (140, 135, 125, 255),

            # Separator
            dpg.mvThemeCol_Separator: (60, 55, 50, 150),
            dpg.mvThemeCol_SeparatorHovered: (90, 82, 72, 255),
            dpg.mvThemeCol_SeparatorActive: (110, 100, 88, 255),

            # Resize grip
            dpg.mvThemeCol_ResizeGrip: (60, 55, 50, 50),
            dpg.mvThemeCol_ResizeGripHovered: (90, 82, 72, 200),
            dpg.mvThemeCol_ResizeGripActive: (110, 100, 88, 255),

            # Plot colors (if we use plots)
            dpg.mvThemeCol_PlotHistogram: (150, 130, 100, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (180, 160, 125, 255),
        },
        "styles": {
            dpg.mvStyleVar_FrameRounding: 6,
            dpg.mvStyleVar_WindowRounding: 8,
            dpg.mvStyleVar_ChildRounding: 6,
            dpg.mvStyleVar_PopupRounding: 6,
            dpg.mvStyleVar_ScrollbarRounding: 8,
            dpg.mvStyleVar_GrabRounding: 4,
            dpg.mvStyleVar_TabRounding: 4,
            dpg.mvStyleVar_FramePadding: (8, 5),
            dpg.mvStyleVar_ItemSpacing: (8, 6),
            dpg.mvStyleVar_WindowPadding: (12, 12),
            dpg.mvStyleVar_FrameBorderSize: 0,
            dpg.mvStyleVar_WindowBorderSize: 0,
        },
    },

    "Sepia Ink": {
        "description": "Warmer sepia tones, like aged paper and ink",
        "colors": {
            # Window/frame backgrounds - warmer sepia
            dpg.mvThemeCol_WindowBg: (38, 34, 30, 255),
            dpg.mvThemeCol_ChildBg: (44, 40, 35, 255),
            dpg.mvThemeCol_PopupBg: (48, 44, 38, 255),
            dpg.mvThemeCol_MenuBarBg: (38, 34, 30, 255),

            # Borders
            dpg.mvThemeCol_Border: (65, 58, 48, 100),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            # Frame backgrounds
            dpg.mvThemeCol_FrameBg: (52, 47, 40, 255),
            dpg.mvThemeCol_FrameBgHovered: (62, 56, 48, 255),
            dpg.mvThemeCol_FrameBgActive: (72, 65, 55, 255),

            # Title bar
            dpg.mvThemeCol_TitleBg: (32, 29, 25, 255),
            dpg.mvThemeCol_TitleBgActive: (44, 40, 35, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (32, 29, 25, 200),

            # Scrollbar
            dpg.mvThemeCol_ScrollbarBg: (38, 34, 30, 255),
            dpg.mvThemeCol_ScrollbarGrab: (75, 68, 55, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (90, 82, 68, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (108, 98, 80, 255),

            # Buttons - rust/terracotta accent
            dpg.mvThemeCol_Button: (85, 65, 55, 255),
            dpg.mvThemeCol_ButtonHovered: (100, 78, 65, 255),
            dpg.mvThemeCol_ButtonActive: (115, 88, 72, 255),

            # Headers
            dpg.mvThemeCol_Header: (58, 52, 44, 255),
            dpg.mvThemeCol_HeaderHovered: (70, 62, 52, 255),
            dpg.mvThemeCol_HeaderActive: (82, 72, 60, 255),

            # Tabs
            dpg.mvThemeCol_Tab: (48, 44, 38, 255),
            dpg.mvThemeCol_TabHovered: (70, 62, 52, 255),
            dpg.mvThemeCol_TabActive: (58, 52, 44, 255),
            dpg.mvThemeCol_TabUnfocused: (44, 40, 35, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (52, 47, 40, 255),

            # Slider grab
            dpg.mvThemeCol_SliderGrab: (140, 115, 85, 255),
            dpg.mvThemeCol_SliderGrabActive: (165, 138, 102, 255),

            # Checkmark
            dpg.mvThemeCol_CheckMark: (200, 170, 130, 255),

            # Text - warm cream
            dpg.mvThemeCol_Text: (245, 238, 220, 255),
            dpg.mvThemeCol_TextDisabled: (150, 140, 125, 255),

            # Separator
            dpg.mvThemeCol_Separator: (65, 58, 48, 150),
            dpg.mvThemeCol_SeparatorHovered: (95, 85, 70, 255),
            dpg.mvThemeCol_SeparatorActive: (115, 102, 82, 255),

            # Resize grip
            dpg.mvThemeCol_ResizeGrip: (65, 58, 48, 50),
            dpg.mvThemeCol_ResizeGripHovered: (95, 85, 70, 200),
            dpg.mvThemeCol_ResizeGripActive: (115, 102, 82, 255),

            # Plot
            dpg.mvThemeCol_PlotHistogram: (165, 135, 95, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (195, 162, 115, 255),
        },
        "styles": {
            dpg.mvStyleVar_FrameRounding: 5,
            dpg.mvStyleVar_WindowRounding: 6,
            dpg.mvStyleVar_ChildRounding: 5,
            dpg.mvStyleVar_PopupRounding: 5,
            dpg.mvStyleVar_ScrollbarRounding: 6,
            dpg.mvStyleVar_GrabRounding: 3,
            dpg.mvStyleVar_TabRounding: 3,
            dpg.mvStyleVar_FramePadding: (8, 5),
            dpg.mvStyleVar_ItemSpacing: (8, 6),
            dpg.mvStyleVar_WindowPadding: (12, 12),
            dpg.mvStyleVar_FrameBorderSize: 0,
            dpg.mvStyleVar_WindowBorderSize: 0,
        },
    },

    "Moss Stone": {
        "description": "Cool-warm balance with mossy green accents",
        "colors": {
            # Window/frame backgrounds - neutral with green undertone
            dpg.mvThemeCol_WindowBg: (32, 34, 32, 255),
            dpg.mvThemeCol_ChildBg: (38, 40, 38, 255),
            dpg.mvThemeCol_PopupBg: (42, 45, 42, 255),
            dpg.mvThemeCol_MenuBarBg: (32, 34, 32, 255),

            # Borders
            dpg.mvThemeCol_Border: (55, 60, 52, 100),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            # Frame backgrounds
            dpg.mvThemeCol_FrameBg: (48, 52, 46, 255),
            dpg.mvThemeCol_FrameBgHovered: (58, 62, 54, 255),
            dpg.mvThemeCol_FrameBgActive: (68, 72, 62, 255),

            # Title bar
            dpg.mvThemeCol_TitleBg: (28, 30, 28, 255),
            dpg.mvThemeCol_TitleBgActive: (38, 40, 38, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (28, 30, 28, 200),

            # Scrollbar
            dpg.mvThemeCol_ScrollbarBg: (32, 34, 32, 255),
            dpg.mvThemeCol_ScrollbarGrab: (68, 75, 62, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (82, 90, 75, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (98, 108, 88, 255),

            # Buttons - moss green accent
            dpg.mvThemeCol_Button: (58, 72, 55, 255),
            dpg.mvThemeCol_ButtonHovered: (70, 88, 65, 255),
            dpg.mvThemeCol_ButtonActive: (82, 102, 75, 255),

            # Headers
            dpg.mvThemeCol_Header: (52, 56, 50, 255),
            dpg.mvThemeCol_HeaderHovered: (62, 68, 58, 255),
            dpg.mvThemeCol_HeaderActive: (72, 78, 66, 255),

            # Tabs
            dpg.mvThemeCol_Tab: (42, 45, 42, 255),
            dpg.mvThemeCol_TabHovered: (62, 68, 58, 255),
            dpg.mvThemeCol_TabActive: (52, 56, 50, 255),
            dpg.mvThemeCol_TabUnfocused: (38, 40, 38, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (48, 52, 46, 255),

            # Slider grab
            dpg.mvThemeCol_SliderGrab: (115, 130, 100, 255),
            dpg.mvThemeCol_SliderGrabActive: (138, 155, 120, 255),

            # Checkmark
            dpg.mvThemeCol_CheckMark: (175, 195, 155, 255),

            # Text - slightly warm white
            dpg.mvThemeCol_Text: (238, 240, 232, 255),
            dpg.mvThemeCol_TextDisabled: (135, 140, 128, 255),

            # Separator
            dpg.mvThemeCol_Separator: (55, 60, 52, 150),
            dpg.mvThemeCol_SeparatorHovered: (85, 95, 78, 255),
            dpg.mvThemeCol_SeparatorActive: (105, 118, 95, 255),

            # Resize grip
            dpg.mvThemeCol_ResizeGrip: (55, 60, 52, 50),
            dpg.mvThemeCol_ResizeGripHovered: (85, 95, 78, 200),
            dpg.mvThemeCol_ResizeGripActive: (105, 118, 95, 255),

            # Plot
            dpg.mvThemeCol_PlotHistogram: (145, 165, 125, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (172, 195, 148, 255),
        },
        "styles": {
            dpg.mvStyleVar_FrameRounding: 4,
            dpg.mvStyleVar_WindowRounding: 6,
            dpg.mvStyleVar_ChildRounding: 4,
            dpg.mvStyleVar_PopupRounding: 4,
            dpg.mvStyleVar_ScrollbarRounding: 6,
            dpg.mvStyleVar_GrabRounding: 3,
            dpg.mvStyleVar_TabRounding: 3,
            dpg.mvStyleVar_FramePadding: (8, 5),
            dpg.mvStyleVar_ItemSpacing: (8, 6),
            dpg.mvStyleVar_WindowPadding: (12, 12),
            dpg.mvStyleVar_FrameBorderSize: 0,
            dpg.mvStyleVar_WindowBorderSize: 0,
        },
    },
}


def _get_display_scale() -> float:
    """Detect display scale factor (Retina = 2.0, normal = 1.0)."""
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
    """Locate DejaVu Sans TTF file from matplotlib's bundled fonts."""
    try:
        import matplotlib
        mpl_data = Path(matplotlib.get_data_path())
        font_path = mpl_data / "fonts" / "ttf" / "DejaVuSans.ttf"
        if font_path.exists():
            return font_path
    except ImportError:
        pass
    return None


# Unicode ranges for math symbols
GREEK_RANGE = (0x0370, 0x03FF)
MATH_OPERATORS_RANGE = (0x2200, 0x22FF)
SUB_SUPER_RANGE = (0x2070, 0x209F)
MATH_SYMBOLS = [0x2202, 0x2207, 0x03C6, 0x2099, 0x00B2, 0x00D7]


def setup_font(font_name: str, size: int = 14) -> int | None:
    """Load and configure a font with math symbol support."""
    if font_name == "DejaVu Sans":
        font_path = _find_dejavu_font()
    else:
        font_path = FONTS.get(font_name)

    if font_path is None or (font_path is not None and not font_path.exists()):
        print(f"Warning: Font {font_name} not found")
        return None

    display_scale = _get_display_scale()
    raster_size = int(size * display_scale)

    # Only DejaVu has full math symbol coverage - for other fonts, just use Latin
    is_dejavu = font_name == "DejaVu Sans"

    with dpg.font_registry():
        with dpg.font(str(font_path), raster_size) as font_id:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            if is_dejavu:
                # DejaVu has extensive Unicode coverage
                dpg.add_font_range(GREEK_RANGE[0], GREEK_RANGE[1])
                dpg.add_font_range(MATH_OPERATORS_RANGE[0], MATH_OPERATORS_RANGE[1])
                dpg.add_font_range(SUB_SUPER_RANGE[0], SUB_SUPER_RANGE[1])
                dpg.add_font_chars(MATH_SYMBOLS)

    dpg.bind_font(font_id)

    if display_scale > 1.0:
        dpg.set_global_font_scale(1.0 / display_scale)

    return font_id


def create_theme(name: str) -> int:
    """Create a DPG theme from definition."""
    theme_def = THEMES[name]

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            for color_id, rgba in theme_def["colors"].items():
                dpg.add_theme_color(color_id, rgba)

            for style_id, value in theme_def["styles"].items():
                if isinstance(value, tuple):
                    dpg.add_theme_style(style_id, value[0], value[1])
                else:
                    dpg.add_theme_style(style_id, value)

    return theme


def apply_theme(theme_name: str):
    """Apply a theme globally."""
    theme_id = theme_ids[theme_name]
    dpg.bind_theme(theme_id)
    dpg.set_value("theme_description", THEMES[theme_name]["description"])
    dpg.set_value("current_theme_label", f"Theme: {theme_name}")


def apply_font(font_name: str):
    """Apply a font (requires restart to take effect in DPG)."""
    dpg.set_value("current_font_label", f"Font: {font_name} (restart to apply)")
    dpg.set_value("pending_font", font_name)


def build_sample_widgets():
    """Build sample widgets to preview theme."""
    with dpg.child_window(width=-1, height=120, border=True):
        dpg.add_text("Sample Panel with Border")
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_button(label="Primary Action")
            dpg.add_button(label="Secondary")
            dpg.add_button(label="Cancel")

        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Enable feature")
            dpg.add_checkbox(label="Another option", default_value=True)

    dpg.add_spacer(height=10)

    dpg.add_text("Sliders and Inputs")
    dpg.add_slider_float(label="Brightness", default_value=0.5, width=200)
    dpg.add_slider_int(label="Iterations", default_value=5, min_value=1, max_value=10, width=200)
    dpg.add_input_float(label="Scale Factor", default_value=1.0, width=200)
    dpg.add_input_text(label="Name", default_value="Untitled", width=200)

    dpg.add_spacer(height=10)

    dpg.add_text("Combo and Radio")
    dpg.add_combo(["Option A", "Option B", "Option C"], label="Select", default_value="Option A", width=200)
    dpg.add_radio_button(["Choice 1", "Choice 2", "Choice 3"], horizontal=True)

    dpg.add_spacer(height=10)

    with dpg.collapsing_header(label="Collapsible Section", default_open=True):
        dpg.add_text("Content inside collapsible header")
        dpg.add_button(label="Nested Button")
        with dpg.tree_node(label="Tree Node"):
            dpg.add_text("Nested content")
            dpg.add_button(label="Deep Button")

    dpg.add_spacer(height=10)

    dpg.add_text("Math symbols: ∂φ/∂n × ∇²φ", color=(200, 195, 185))

    dpg.add_spacer(height=10)

    dpg.add_text("Disabled Elements", color=(140, 135, 125))
    dpg.add_button(label="Disabled Button", enabled=False)
    dpg.add_slider_float(label="Disabled Slider", enabled=False, default_value=0.3, width=200)


def main():
    global theme_ids

    # Check command line for font selection
    initial_font = "DejaVu Sans"
    if len(sys.argv) > 1:
        requested = sys.argv[1]
        if requested in FONTS:
            initial_font = requested
        else:
            print(f"Unknown font: {requested}")
            print(f"Available: {', '.join(FONTS.keys())}")
            sys.exit(1)

    dpg.create_context()

    # Setup font (must be before any widgets)
    setup_font(initial_font, size=14)

    # Create all themes
    theme_ids = {name: create_theme(name) for name in THEMES}

    # Create viewport
    dpg.create_viewport(title="Elliptica Theme Preview", width=600, height=800)

    with dpg.window(label="Theme Preview", tag="primary_window"):
        # Font selection
        dpg.add_text("Font Selection", color=(180, 175, 165))
        with dpg.group(horizontal=True):
            for name in FONTS:
                dpg.add_button(
                    label=name,
                    callback=lambda s, a, u: apply_font(u),
                    user_data=name,
                )
        dpg.add_text(f"Font: {initial_font}", tag="current_font_label")
        dpg.add_text("", tag="pending_font", show=False)

        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Theme selection
        dpg.add_text("Theme Selection", color=(180, 175, 165))
        with dpg.group(horizontal=True):
            for name in THEMES:
                dpg.add_button(
                    label=name,
                    callback=lambda s, a, u: apply_theme(u),
                    user_data=name,
                )

        dpg.add_text("", tag="current_theme_label")
        dpg.add_text("", tag="theme_description", color=(160, 155, 145))

        dpg.add_separator()
        dpg.add_spacer(height=10)

        build_sample_widgets()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    # Apply first theme
    apply_theme(list(THEMES.keys())[0])

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    print("Usage: python tools/theme_preview.py [font_name]")
    print(f"Available fonts: {', '.join(FONTS.keys())}")
    print()
    main()
