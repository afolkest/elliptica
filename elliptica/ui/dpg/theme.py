"""Steel Blue Soft theme for Elliptica UI."""

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None


def apply_theme() -> None:
    """Apply the Steel Blue Soft theme globally."""
    if dpg is None:
        return

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            # Window/frame backgrounds
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (32, 34, 45, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (38, 40, 50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (44, 46, 56, 255))
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (32, 34, 45, 255))

            # Borders
            dpg.add_theme_color(dpg.mvThemeCol_Border, (62, 64, 74, 80))
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, (0, 0, 0, 0))

            # Frame backgrounds (inputs, sliders) - needs good contrast with ChildBg
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (56, 60, 72, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (66, 70, 82, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (76, 80, 92, 255))

            # Title bar
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (26, 28, 38, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (36, 38, 48, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, (26, 28, 38, 200))

            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (32, 34, 45, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (66, 72, 82, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (76, 82, 94, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (86, 92, 106, 255))

            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, (64, 70, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (74, 80, 92, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (84, 90, 104, 255))

            # Headers
            dpg.add_theme_color(dpg.mvThemeCol_Header, (52, 54, 64, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (62, 64, 74, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (72, 74, 84, 255))

            # Tabs
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (46, 48, 58, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (60, 62, 72, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (54, 56, 66, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, (42, 44, 54, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, (50, 52, 62, 255))

            # Slider grab
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (100, 108, 122, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (116, 124, 140, 255))

            # Checkmark
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (150, 160, 176, 255))

            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, (218, 222, 228, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (134, 138, 144, 255))

            # Separator
            dpg.add_theme_color(dpg.mvThemeCol_Separator, (62, 64, 74, 100))
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, (88, 90, 100, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, (106, 108, 118, 255))

            # Resize grip
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip, (62, 64, 74, 45))
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered, (88, 90, 100, 180))
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive, (106, 108, 118, 255))

            # Plot
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (112, 122, 140, 255))
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogramHovered, (132, 142, 162, 255))

            # Text selection and modal dim
            dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, (80, 95, 115, 180))
            dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg, (0, 0, 0, 140))

            # Rounding (2.5px)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2.5)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 3.0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 2.5)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 2.5)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 2.5)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 2.5)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 2.5)

            # Padding and spacing
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)

    dpg.bind_theme(theme)
