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

# Available fonts (just DejaVu for now - best math symbol coverage)
FONTS = {
    "DejaVu Sans": None,  # Will use matplotlib's bundled font
}

# Theme definitions - Japanese-inspired dark aesthetics
# References: https://pigment.tokyo/en/blogs/article/colors-of-sumi
#             https://artistpigments.org/experiments/traditional_japanese_color_names
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

    "Soft Umber": {
        "description": "Warm brown-gray, very muted rose-brown accent, gentle",
        "colors": {
            # Window/frame backgrounds - warm umber gray
            dpg.mvThemeCol_WindowBg: (34, 32, 31, 255),
            dpg.mvThemeCol_ChildBg: (40, 38, 36, 255),
            dpg.mvThemeCol_PopupBg: (46, 43, 41, 255),
            dpg.mvThemeCol_MenuBarBg: (34, 32, 31, 255),

            # Borders - very subtle
            dpg.mvThemeCol_Border: (55, 52, 48, 70),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            # Frame backgrounds
            dpg.mvThemeCol_FrameBg: (46, 43, 40, 255),
            dpg.mvThemeCol_FrameBgHovered: (54, 50, 47, 255),
            dpg.mvThemeCol_FrameBgActive: (62, 58, 54, 255),

            # Title bar
            dpg.mvThemeCol_TitleBg: (30, 28, 27, 255),
            dpg.mvThemeCol_TitleBgActive: (40, 38, 36, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (30, 28, 27, 200),

            # Scrollbar
            dpg.mvThemeCol_ScrollbarBg: (34, 32, 31, 255),
            dpg.mvThemeCol_ScrollbarGrab: (62, 58, 54, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (75, 70, 65, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (88, 82, 76, 255),

            # Buttons - muted rose-brown (barely distinguishable)
            dpg.mvThemeCol_Button: (58, 52, 52, 255),
            dpg.mvThemeCol_ButtonHovered: (70, 62, 62, 255),
            dpg.mvThemeCol_ButtonActive: (82, 72, 72, 255),

            # Headers
            dpg.mvThemeCol_Header: (50, 47, 44, 255),
            dpg.mvThemeCol_HeaderHovered: (60, 56, 52, 255),
            dpg.mvThemeCol_HeaderActive: (70, 65, 60, 255),

            # Tabs
            dpg.mvThemeCol_Tab: (42, 40, 38, 255),
            dpg.mvThemeCol_TabHovered: (56, 52, 49, 255),
            dpg.mvThemeCol_TabActive: (50, 47, 44, 255),
            dpg.mvThemeCol_TabUnfocused: (38, 36, 34, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (46, 43, 40, 255),

            # Slider grab
            dpg.mvThemeCol_SliderGrab: (95, 88, 82, 255),
            dpg.mvThemeCol_SliderGrabActive: (115, 106, 98, 255),

            # Checkmark - muted warm
            dpg.mvThemeCol_CheckMark: (155, 140, 130, 255),

            # Text - soft cream
            dpg.mvThemeCol_Text: (235, 230, 222, 255),
            dpg.mvThemeCol_TextDisabled: (130, 125, 118, 255),

            # Separator
            dpg.mvThemeCol_Separator: (55, 52, 48, 100),
            dpg.mvThemeCol_SeparatorHovered: (78, 72, 66, 255),
            dpg.mvThemeCol_SeparatorActive: (95, 88, 80, 255),

            # Resize grip
            dpg.mvThemeCol_ResizeGrip: (55, 52, 48, 40),
            dpg.mvThemeCol_ResizeGripHovered: (78, 72, 66, 160),
            dpg.mvThemeCol_ResizeGripActive: (95, 88, 80, 255),

            # Plot
            dpg.mvThemeCol_PlotHistogram: (130, 118, 108, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (155, 140, 128, 255),
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

    "Rikyū 利休": {
        "description": "Tea master - muted olive-gray (利休鼠), straw gold accent, extreme wabi-sabi",
        "colors": {
            # Window/frame backgrounds - rikyū-nezumi (gray with subtle olive)
            dpg.mvThemeCol_WindowBg: (30, 31, 28, 255),
            dpg.mvThemeCol_ChildBg: (36, 38, 34, 255),
            dpg.mvThemeCol_PopupBg: (42, 44, 40, 255),
            dpg.mvThemeCol_MenuBarBg: (30, 31, 28, 255),

            # Borders - very subtle, like aged wood grain
            dpg.mvThemeCol_Border: (52, 55, 48, 70),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            # Frame backgrounds
            dpg.mvThemeCol_FrameBg: (42, 44, 40, 255),
            dpg.mvThemeCol_FrameBgHovered: (50, 53, 47, 255),
            dpg.mvThemeCol_FrameBgActive: (58, 62, 55, 255),

            # Title bar
            dpg.mvThemeCol_TitleBg: (26, 27, 24, 255),
            dpg.mvThemeCol_TitleBgActive: (36, 38, 34, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (26, 27, 24, 200),

            # Scrollbar
            dpg.mvThemeCol_ScrollbarBg: (30, 31, 28, 255),
            dpg.mvThemeCol_ScrollbarGrab: (58, 62, 52, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (72, 78, 65, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (88, 95, 78, 255),

            # Buttons - muted straw/bamboo gold (very understated)
            dpg.mvThemeCol_Button: (70, 68, 55, 255),
            dpg.mvThemeCol_ButtonHovered: (85, 82, 65, 255),
            dpg.mvThemeCol_ButtonActive: (100, 96, 75, 255),

            # Headers
            dpg.mvThemeCol_Header: (48, 50, 44, 255),
            dpg.mvThemeCol_HeaderHovered: (58, 60, 52, 255),
            dpg.mvThemeCol_HeaderActive: (68, 72, 62, 255),

            # Tabs
            dpg.mvThemeCol_Tab: (38, 40, 35, 255),
            dpg.mvThemeCol_TabHovered: (52, 55, 48, 255),
            dpg.mvThemeCol_TabActive: (48, 50, 44, 255),
            dpg.mvThemeCol_TabUnfocused: (34, 35, 31, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (42, 44, 40, 255),

            # Slider grab - aged gold
            dpg.mvThemeCol_SliderGrab: (125, 118, 90, 255),
            dpg.mvThemeCol_SliderGrabActive: (150, 142, 108, 255),

            # Checkmark - muted gold
            dpg.mvThemeCol_CheckMark: (165, 155, 115, 255),

            # Text - soft warm white (like old paper)
            dpg.mvThemeCol_Text: (232, 228, 218, 255),
            dpg.mvThemeCol_TextDisabled: (125, 122, 112, 255),

            # Separator
            dpg.mvThemeCol_Separator: (52, 55, 48, 100),
            dpg.mvThemeCol_SeparatorHovered: (78, 82, 70, 255),
            dpg.mvThemeCol_SeparatorActive: (98, 105, 88, 255),

            # Resize grip
            dpg.mvThemeCol_ResizeGrip: (52, 55, 48, 40),
            dpg.mvThemeCol_ResizeGripHovered: (78, 82, 70, 160),
            dpg.mvThemeCol_ResizeGripActive: (98, 105, 88, 255),

            # Plot
            dpg.mvThemeCol_PlotHistogram: (145, 138, 105, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (175, 165, 125, 255),
        },
        "styles": {
            dpg.mvStyleVar_FrameRounding: 4,
            dpg.mvStyleVar_WindowRounding: 5,
            dpg.mvStyleVar_ChildRounding: 4,
            dpg.mvStyleVar_PopupRounding: 4,
            dpg.mvStyleVar_ScrollbarRounding: 5,
            dpg.mvStyleVar_GrabRounding: 3,
            dpg.mvStyleVar_TabRounding: 3,
            dpg.mvStyleVar_FramePadding: (9, 5),
            dpg.mvStyleVar_ItemSpacing: (9, 7),
            dpg.mvStyleVar_WindowPadding: (14, 14),
            dpg.mvStyleVar_FrameBorderSize: 0,
            dpg.mvStyleVar_WindowBorderSize: 0,
        },
    },

    "Quiet Stone": {
        "description": "Cool-neutral gray, hint of blue-gray accent, monastic calm",
        "colors": {
            # Window/frame backgrounds - neutral with slight cool undertone
            dpg.mvThemeCol_WindowBg: (33, 34, 35, 255),
            dpg.mvThemeCol_ChildBg: (39, 40, 42, 255),
            dpg.mvThemeCol_PopupBg: (45, 46, 48, 255),
            dpg.mvThemeCol_MenuBarBg: (33, 34, 35, 255),

            # Borders
            dpg.mvThemeCol_Border: (52, 54, 56, 70),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            # Frame backgrounds
            dpg.mvThemeCol_FrameBg: (45, 46, 48, 255),
            dpg.mvThemeCol_FrameBgHovered: (52, 54, 56, 255),
            dpg.mvThemeCol_FrameBgActive: (60, 62, 65, 255),

            # Title bar
            dpg.mvThemeCol_TitleBg: (29, 30, 31, 255),
            dpg.mvThemeCol_TitleBgActive: (39, 40, 42, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (29, 30, 31, 200),

            # Scrollbar
            dpg.mvThemeCol_ScrollbarBg: (33, 34, 35, 255),
            dpg.mvThemeCol_ScrollbarGrab: (58, 60, 64, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (72, 74, 78, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (85, 88, 92, 255),

            # Buttons - barely tinted blue-gray
            dpg.mvThemeCol_Button: (52, 54, 58, 255),
            dpg.mvThemeCol_ButtonHovered: (62, 65, 70, 255),
            dpg.mvThemeCol_ButtonActive: (72, 75, 82, 255),

            # Headers
            dpg.mvThemeCol_Header: (48, 50, 52, 255),
            dpg.mvThemeCol_HeaderHovered: (56, 58, 62, 255),
            dpg.mvThemeCol_HeaderActive: (65, 68, 72, 255),

            # Tabs
            dpg.mvThemeCol_Tab: (40, 42, 44, 255),
            dpg.mvThemeCol_TabHovered: (52, 54, 58, 255),
            dpg.mvThemeCol_TabActive: (48, 50, 52, 255),
            dpg.mvThemeCol_TabUnfocused: (36, 38, 40, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (45, 46, 48, 255),

            # Slider grab
            dpg.mvThemeCol_SliderGrab: (92, 96, 102, 255),
            dpg.mvThemeCol_SliderGrabActive: (112, 116, 124, 255),

            # Checkmark
            dpg.mvThemeCol_CheckMark: (150, 155, 165, 255),

            # Text - slightly warm white (not cold)
            dpg.mvThemeCol_Text: (232, 230, 226, 255),
            dpg.mvThemeCol_TextDisabled: (125, 125, 122, 255),

            # Separator
            dpg.mvThemeCol_Separator: (52, 54, 56, 100),
            dpg.mvThemeCol_SeparatorHovered: (75, 78, 82, 255),
            dpg.mvThemeCol_SeparatorActive: (92, 96, 102, 255),

            # Resize grip
            dpg.mvThemeCol_ResizeGrip: (52, 54, 56, 40),
            dpg.mvThemeCol_ResizeGripHovered: (75, 78, 82, 160),
            dpg.mvThemeCol_ResizeGripActive: (92, 96, 102, 255),

            # Plot
            dpg.mvThemeCol_PlotHistogram: (125, 130, 140, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (150, 155, 168, 255),
        },
        "styles": {
            dpg.mvStyleVar_FrameRounding: 4,
            dpg.mvStyleVar_WindowRounding: 5,
            dpg.mvStyleVar_ChildRounding: 4,
            dpg.mvStyleVar_PopupRounding: 4,
            dpg.mvStyleVar_ScrollbarRounding: 5,
            dpg.mvStyleVar_GrabRounding: 3,
            dpg.mvStyleVar_TabRounding: 3,
            dpg.mvStyleVar_FramePadding: (8, 5),
            dpg.mvStyleVar_ItemSpacing: (8, 6),
            dpg.mvStyleVar_WindowPadding: (12, 12),
            dpg.mvStyleVar_FrameBorderSize: 0,
            dpg.mvStyleVar_WindowBorderSize: 0,
        },
    },

    "Deep Loam": {
        "description": "Earthy dark brown, no accent color at all, purely tonal",
        "colors": {
            # Window/frame backgrounds - deep earthy brown
            dpg.mvThemeCol_WindowBg: (32, 30, 28, 255),
            dpg.mvThemeCol_ChildBg: (38, 36, 33, 255),
            dpg.mvThemeCol_PopupBg: (44, 41, 38, 255),
            dpg.mvThemeCol_MenuBarBg: (32, 30, 28, 255),

            # Borders
            dpg.mvThemeCol_Border: (52, 48, 44, 70),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            # Frame backgrounds
            dpg.mvThemeCol_FrameBg: (44, 41, 38, 255),
            dpg.mvThemeCol_FrameBgHovered: (52, 48, 44, 255),
            dpg.mvThemeCol_FrameBgActive: (60, 56, 51, 255),

            # Title bar
            dpg.mvThemeCol_TitleBg: (28, 26, 24, 255),
            dpg.mvThemeCol_TitleBgActive: (38, 36, 33, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (28, 26, 24, 200),

            # Scrollbar
            dpg.mvThemeCol_ScrollbarBg: (32, 30, 28, 255),
            dpg.mvThemeCol_ScrollbarGrab: (58, 54, 49, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (72, 66, 60, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (85, 78, 70, 255),

            # Buttons - same family, just slightly lighter (no hue shift)
            dpg.mvThemeCol_Button: (52, 48, 44, 255),
            dpg.mvThemeCol_ButtonHovered: (62, 57, 52, 255),
            dpg.mvThemeCol_ButtonActive: (72, 66, 60, 255),

            # Headers
            dpg.mvThemeCol_Header: (48, 44, 40, 255),
            dpg.mvThemeCol_HeaderHovered: (56, 52, 47, 255),
            dpg.mvThemeCol_HeaderActive: (65, 60, 54, 255),

            # Tabs
            dpg.mvThemeCol_Tab: (40, 37, 34, 255),
            dpg.mvThemeCol_TabHovered: (52, 48, 44, 255),
            dpg.mvThemeCol_TabActive: (48, 44, 40, 255),
            dpg.mvThemeCol_TabUnfocused: (36, 34, 31, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (44, 41, 38, 255),

            # Slider grab
            dpg.mvThemeCol_SliderGrab: (90, 82, 74, 255),
            dpg.mvThemeCol_SliderGrabActive: (110, 100, 90, 255),

            # Checkmark
            dpg.mvThemeCol_CheckMark: (148, 138, 125, 255),

            # Text - warm parchment
            dpg.mvThemeCol_Text: (235, 228, 218, 255),
            dpg.mvThemeCol_TextDisabled: (128, 120, 112, 255),

            # Separator
            dpg.mvThemeCol_Separator: (52, 48, 44, 100),
            dpg.mvThemeCol_SeparatorHovered: (75, 68, 62, 255),
            dpg.mvThemeCol_SeparatorActive: (92, 84, 75, 255),

            # Resize grip
            dpg.mvThemeCol_ResizeGrip: (52, 48, 44, 40),
            dpg.mvThemeCol_ResizeGripHovered: (75, 68, 62, 160),
            dpg.mvThemeCol_ResizeGripActive: (92, 84, 75, 255),

            # Plot
            dpg.mvThemeCol_PlotHistogram: (125, 115, 102, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (150, 138, 122, 255),
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

    "Sumi Rust": {
        "description": "Charcoal sumi paper with rusted iron accents; warm, grounded, non-tech",
        "colors": {
            dpg.mvThemeCol_WindowBg: (28, 26, 25, 255),
            dpg.mvThemeCol_ChildBg: (34, 32, 31, 255),
            dpg.mvThemeCol_PopupBg: (40, 38, 36, 255),
            dpg.mvThemeCol_MenuBarBg: (28, 26, 25, 255),

            dpg.mvThemeCol_Border: (55, 50, 45, 70),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (44, 41, 38, 255),
            dpg.mvThemeCol_FrameBgHovered: (52, 48, 45, 255),
            dpg.mvThemeCol_FrameBgActive: (60, 55, 50, 255),

            dpg.mvThemeCol_TitleBg: (24, 22, 21, 255),
            dpg.mvThemeCol_TitleBgActive: (34, 32, 31, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (24, 22, 21, 200),

            dpg.mvThemeCol_ScrollbarBg: (28, 26, 25, 255),
            dpg.mvThemeCol_ScrollbarGrab: (68, 55, 46, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (82, 65, 55, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (98, 78, 64, 255),

            dpg.mvThemeCol_Button: (70, 54, 45, 255),
            dpg.mvThemeCol_ButtonHovered: (84, 65, 54, 255),
            dpg.mvThemeCol_ButtonActive: (98, 78, 64, 255),

            dpg.mvThemeCol_Header: (48, 44, 41, 255),
            dpg.mvThemeCol_HeaderHovered: (58, 52, 48, 255),
            dpg.mvThemeCol_HeaderActive: (68, 60, 54, 255),

            dpg.mvThemeCol_Tab: (40, 37, 35, 255),
            dpg.mvThemeCol_TabHovered: (54, 48, 44, 255),
            dpg.mvThemeCol_TabActive: (48, 44, 41, 255),
            dpg.mvThemeCol_TabUnfocused: (34, 31, 30, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (42, 38, 36, 255),

            dpg.mvThemeCol_SliderGrab: (120, 95, 80, 255),
            dpg.mvThemeCol_SliderGrabActive: (142, 112, 95, 255),

            dpg.mvThemeCol_CheckMark: (175, 145, 120, 255),

            dpg.mvThemeCol_Text: (236, 228, 218, 255),
            dpg.mvThemeCol_TextDisabled: (135, 128, 118, 255),

            dpg.mvThemeCol_Separator: (55, 50, 45, 110),
            dpg.mvThemeCol_SeparatorHovered: (82, 70, 62, 255),
            dpg.mvThemeCol_SeparatorActive: (102, 86, 74, 255),

            dpg.mvThemeCol_ResizeGrip: (55, 50, 45, 40),
            dpg.mvThemeCol_ResizeGripHovered: (82, 70, 62, 170),
            dpg.mvThemeCol_ResizeGripActive: (102, 86, 74, 255),

            dpg.mvThemeCol_PlotHistogram: (138, 118, 104, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (165, 142, 124, 255),
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

    "Cedar Night": {
        "description": "Dark cedar plank with moss and candlewax; cozy studio vibe",
        "colors": {
            dpg.mvThemeCol_WindowBg: (27, 24, 22, 255),
            dpg.mvThemeCol_ChildBg: (33, 30, 27, 255),
            dpg.mvThemeCol_PopupBg: (39, 35, 32, 255),
            dpg.mvThemeCol_MenuBarBg: (27, 24, 22, 255),

            dpg.mvThemeCol_Border: (54, 46, 40, 70),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (42, 38, 34, 255),
            dpg.mvThemeCol_FrameBgHovered: (50, 45, 40, 255),
            dpg.mvThemeCol_FrameBgActive: (60, 53, 48, 255),

            dpg.mvThemeCol_TitleBg: (23, 21, 19, 255),
            dpg.mvThemeCol_TitleBgActive: (33, 30, 27, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (23, 21, 19, 200),

            dpg.mvThemeCol_ScrollbarBg: (27, 24, 22, 255),
            dpg.mvThemeCol_ScrollbarGrab: (64, 60, 48, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (76, 72, 56, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (92, 86, 66, 255),

            dpg.mvThemeCol_Button: (60, 54, 46, 255),
            dpg.mvThemeCol_ButtonHovered: (72, 66, 56, 255),
            dpg.mvThemeCol_ButtonActive: (86, 78, 66, 255),

            dpg.mvThemeCol_Header: (46, 42, 38, 255),
            dpg.mvThemeCol_HeaderHovered: (56, 50, 45, 255),
            dpg.mvThemeCol_HeaderActive: (66, 58, 50, 255),

            dpg.mvThemeCol_Tab: (38, 34, 31, 255),
            dpg.mvThemeCol_TabHovered: (52, 46, 41, 255),
            dpg.mvThemeCol_TabActive: (46, 42, 38, 255),
            dpg.mvThemeCol_TabUnfocused: (32, 29, 27, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (40, 36, 33, 255),

            dpg.mvThemeCol_SliderGrab: (110, 100, 78, 255),
            dpg.mvThemeCol_SliderGrabActive: (132, 120, 92, 255),

            dpg.mvThemeCol_CheckMark: (160, 150, 115, 255),

            dpg.mvThemeCol_Text: (235, 226, 214, 255),
            dpg.mvThemeCol_TextDisabled: (132, 124, 114, 255),

            dpg.mvThemeCol_Separator: (54, 46, 40, 110),
            dpg.mvThemeCol_SeparatorHovered: (78, 70, 60, 255),
            dpg.mvThemeCol_SeparatorActive: (96, 86, 72, 255),

            dpg.mvThemeCol_ResizeGrip: (54, 46, 40, 40),
            dpg.mvThemeCol_ResizeGripHovered: (78, 70, 60, 170),
            dpg.mvThemeCol_ResizeGripActive: (96, 86, 72, 255),

            dpg.mvThemeCol_PlotHistogram: (130, 116, 94, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (156, 138, 110, 255),
        },
        "styles": {
            dpg.mvStyleVar_FrameRounding: 6,
            dpg.mvStyleVar_WindowRounding: 7,
            dpg.mvStyleVar_ChildRounding: 5,
            dpg.mvStyleVar_PopupRounding: 6,
            dpg.mvStyleVar_ScrollbarRounding: 7,
            dpg.mvStyleVar_GrabRounding: 3,
            dpg.mvStyleVar_TabRounding: 3,
            dpg.mvStyleVar_FramePadding: (9, 5),
            dpg.mvStyleVar_ItemSpacing: (9, 6),
            dpg.mvStyleVar_WindowPadding: (13, 12),
            dpg.mvStyleVar_FrameBorderSize: 0,
            dpg.mvStyleVar_WindowBorderSize: 0,
        },
    },

    "Rose Ash": {
        "description": "Smoky gray-brown with muted rosewood highlights; soft and painterly",
        "colors": {
            dpg.mvThemeCol_WindowBg: (31, 28, 29, 255),
            dpg.mvThemeCol_ChildBg: (37, 34, 35, 255),
            dpg.mvThemeCol_PopupBg: (43, 39, 40, 255),
            dpg.mvThemeCol_MenuBarBg: (31, 28, 29, 255),

            dpg.mvThemeCol_Border: (56, 50, 52, 80),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (45, 41, 42, 255),
            dpg.mvThemeCol_FrameBgHovered: (54, 48, 50, 255),
            dpg.mvThemeCol_FrameBgActive: (64, 57, 58, 255),

            dpg.mvThemeCol_TitleBg: (27, 24, 25, 255),
            dpg.mvThemeCol_TitleBgActive: (37, 34, 35, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (27, 24, 25, 200),

            dpg.mvThemeCol_ScrollbarBg: (31, 28, 29, 255),
            dpg.mvThemeCol_ScrollbarGrab: (76, 62, 66, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (92, 76, 80, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (108, 90, 92, 255),

            dpg.mvThemeCol_Button: (74, 60, 64, 255),
            dpg.mvThemeCol_ButtonHovered: (88, 72, 76, 255),
            dpg.mvThemeCol_ButtonActive: (104, 86, 88, 255),

            dpg.mvThemeCol_Header: (50, 46, 47, 255),
            dpg.mvThemeCol_HeaderHovered: (60, 54, 55, 255),
            dpg.mvThemeCol_HeaderActive: (70, 62, 62, 255),

            dpg.mvThemeCol_Tab: (42, 38, 39, 255),
            dpg.mvThemeCol_TabHovered: (56, 50, 51, 255),
            dpg.mvThemeCol_TabActive: (50, 46, 47, 255),
            dpg.mvThemeCol_TabUnfocused: (36, 32, 33, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (44, 40, 41, 255),

            dpg.mvThemeCol_SliderGrab: (126, 104, 110, 255),
            dpg.mvThemeCol_SliderGrabActive: (148, 122, 126, 255),

            dpg.mvThemeCol_CheckMark: (186, 160, 166, 255),

            dpg.mvThemeCol_Text: (240, 232, 226, 255),
            dpg.mvThemeCol_TextDisabled: (138, 130, 126, 255),

            dpg.mvThemeCol_Separator: (56, 50, 52, 120),
            dpg.mvThemeCol_SeparatorHovered: (86, 74, 78, 255),
            dpg.mvThemeCol_SeparatorActive: (104, 90, 94, 255),

            dpg.mvThemeCol_ResizeGrip: (56, 50, 52, 45),
            dpg.mvThemeCol_ResizeGripHovered: (86, 74, 78, 180),
            dpg.mvThemeCol_ResizeGripActive: (104, 90, 94, 255),

            dpg.mvThemeCol_PlotHistogram: (142, 124, 128, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (166, 146, 150, 255),
        },
        "styles": {
            dpg.mvStyleVar_FrameRounding: 5,
            dpg.mvStyleVar_WindowRounding: 6,
            dpg.mvStyleVar_ChildRounding: 4,
            dpg.mvStyleVar_PopupRounding: 5,
            dpg.mvStyleVar_ScrollbarRounding: 6,
            dpg.mvStyleVar_GrabRounding: 3,
            dpg.mvStyleVar_TabRounding: 3,
            dpg.mvStyleVar_FramePadding: (9, 5),
            dpg.mvStyleVar_ItemSpacing: (9, 6),
            dpg.mvStyleVar_WindowPadding: (12, 12),
            dpg.mvStyleVar_FrameBorderSize: 0,
            dpg.mvStyleVar_WindowBorderSize: 0,
        },
    },

    "Dusty Fig": {
        "description": "Faded fig skin and clay dust, restrained purples that stay warm",
        "colors": {
            dpg.mvThemeCol_WindowBg: (29, 26, 30, 255),
            dpg.mvThemeCol_ChildBg: (35, 32, 36, 255),
            dpg.mvThemeCol_PopupBg: (41, 38, 42, 255),
            dpg.mvThemeCol_MenuBarBg: (29, 26, 30, 255),

            dpg.mvThemeCol_Border: (58, 52, 60, 80),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (43, 39, 45, 255),
            dpg.mvThemeCol_FrameBgHovered: (52, 46, 53, 255),
            dpg.mvThemeCol_FrameBgActive: (62, 54, 62, 255),

            dpg.mvThemeCol_TitleBg: (25, 23, 26, 255),
            dpg.mvThemeCol_TitleBgActive: (35, 32, 36, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (25, 23, 26, 200),

            dpg.mvThemeCol_ScrollbarBg: (29, 26, 30, 255),
            dpg.mvThemeCol_ScrollbarGrab: (74, 62, 76, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (88, 74, 90, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (104, 88, 104, 255),

            dpg.mvThemeCol_Button: (70, 58, 72, 255),
            dpg.mvThemeCol_ButtonHovered: (84, 70, 86, 255),
            dpg.mvThemeCol_ButtonActive: (100, 84, 100, 255),

            dpg.mvThemeCol_Header: (48, 44, 50, 255),
            dpg.mvThemeCol_HeaderHovered: (58, 52, 60, 255),
            dpg.mvThemeCol_HeaderActive: (68, 60, 70, 255),

            dpg.mvThemeCol_Tab: (40, 36, 42, 255),
            dpg.mvThemeCol_TabHovered: (54, 48, 56, 255),
            dpg.mvThemeCol_TabActive: (48, 44, 50, 255),
            dpg.mvThemeCol_TabUnfocused: (34, 31, 36, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (42, 38, 44, 255),

            dpg.mvThemeCol_SliderGrab: (128, 108, 128, 255),
            dpg.mvThemeCol_SliderGrabActive: (150, 126, 148, 255),

            dpg.mvThemeCol_CheckMark: (190, 166, 184, 255),

            dpg.mvThemeCol_Text: (240, 232, 226, 255),
            dpg.mvThemeCol_TextDisabled: (140, 132, 128, 255),

            dpg.mvThemeCol_Separator: (58, 52, 60, 120),
            dpg.mvThemeCol_SeparatorHovered: (86, 76, 90, 255),
            dpg.mvThemeCol_SeparatorActive: (106, 94, 112, 255),

            dpg.mvThemeCol_ResizeGrip: (58, 52, 60, 45),
            dpg.mvThemeCol_ResizeGripHovered: (86, 76, 90, 180),
            dpg.mvThemeCol_ResizeGripActive: (106, 94, 112, 255),

            dpg.mvThemeCol_PlotHistogram: (144, 126, 146, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (170, 148, 170, 255),
        },
        "styles": {
            dpg.mvStyleVar_FrameRounding: 5,
            dpg.mvStyleVar_WindowRounding: 6,
            dpg.mvStyleVar_ChildRounding: 4,
            dpg.mvStyleVar_PopupRounding: 5,
            dpg.mvStyleVar_ScrollbarRounding: 6,
            dpg.mvStyleVar_GrabRounding: 3,
            dpg.mvStyleVar_TabRounding: 3,
            dpg.mvStyleVar_FramePadding: (8, 5),
            dpg.mvStyleVar_ItemSpacing: (9, 6),
            dpg.mvStyleVar_WindowPadding: (12, 12),
            dpg.mvStyleVar_FrameBorderSize: 0,
            dpg.mvStyleVar_WindowBorderSize: 0,
        },
    },

    "Brass Graphite": {
        "description": "Graphite charcoal with clear brass accents for focus guidance",
        "colors": {
            dpg.mvThemeCol_WindowBg: (24, 24, 25, 255),
            dpg.mvThemeCol_ChildBg: (30, 30, 32, 255),
            dpg.mvThemeCol_PopupBg: (36, 36, 38, 255),
            dpg.mvThemeCol_MenuBarBg: (24, 24, 25, 255),

            dpg.mvThemeCol_Border: (60, 58, 52, 90),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (42, 42, 44, 255),
            dpg.mvThemeCol_FrameBgHovered: (52, 52, 54, 255),
            dpg.mvThemeCol_FrameBgActive: (62, 62, 64, 255),

            dpg.mvThemeCol_TitleBg: (20, 20, 21, 255),
            dpg.mvThemeCol_TitleBgActive: (30, 30, 32, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (20, 20, 21, 200),

            dpg.mvThemeCol_ScrollbarBg: (24, 24, 25, 255),
            dpg.mvThemeCol_ScrollbarGrab: (96, 82, 58, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (116, 98, 70, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (136, 118, 82, 255),

            dpg.mvThemeCol_Button: (96, 82, 58, 255),
            dpg.mvThemeCol_ButtonHovered: (116, 98, 70, 255),
            dpg.mvThemeCol_ButtonActive: (136, 118, 82, 255),

            dpg.mvThemeCol_Header: (52, 50, 48, 255),
            dpg.mvThemeCol_HeaderHovered: (64, 60, 56, 255),
            dpg.mvThemeCol_HeaderActive: (76, 72, 66, 255),

            dpg.mvThemeCol_Tab: (40, 40, 42, 255),
            dpg.mvThemeCol_TabHovered: (56, 54, 52, 255),
            dpg.mvThemeCol_TabActive: (48, 48, 50, 255),
            dpg.mvThemeCol_TabUnfocused: (34, 34, 36, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (42, 42, 44, 255),

            dpg.mvThemeCol_SliderGrab: (158, 138, 102, 255),
            dpg.mvThemeCol_SliderGrabActive: (186, 160, 120, 255),

            dpg.mvThemeCol_CheckMark: (205, 175, 125, 255),

            dpg.mvThemeCol_Text: (242, 236, 224, 255),
            dpg.mvThemeCol_TextDisabled: (140, 134, 124, 255),

            dpg.mvThemeCol_Separator: (60, 58, 52, 120),
            dpg.mvThemeCol_SeparatorHovered: (90, 82, 70, 255),
            dpg.mvThemeCol_SeparatorActive: (110, 100, 84, 255),

            dpg.mvThemeCol_ResizeGrip: (60, 58, 52, 50),
            dpg.mvThemeCol_ResizeGripHovered: (90, 82, 70, 190),
            dpg.mvThemeCol_ResizeGripActive: (110, 100, 84, 255),

            dpg.mvThemeCol_PlotHistogram: (150, 132, 102, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (176, 156, 124, 255),
        },
        "styles": {
            dpg.mvStyleVar_FrameRounding: 5,
            dpg.mvStyleVar_WindowRounding: 6,
            dpg.mvStyleVar_ChildRounding: 4,
            dpg.mvStyleVar_PopupRounding: 5,
            dpg.mvStyleVar_ScrollbarRounding: 6,
            dpg.mvStyleVar_GrabRounding: 3,
            dpg.mvStyleVar_TabRounding: 3,
            dpg.mvStyleVar_FramePadding: (8, 5),
            dpg.mvStyleVar_ItemSpacing: (9, 6),
            dpg.mvStyleVar_WindowPadding: (12, 12),
            dpg.mvStyleVar_FrameBorderSize: 0,
            dpg.mvStyleVar_WindowBorderSize: 0,
        },
    },

    "High-Contrast Parchment": {
        "description": "Ink-on-parchment contrast: crisp cream text, amber focus cues",
        "colors": {
            dpg.mvThemeCol_WindowBg: (18, 17, 16, 255),
            dpg.mvThemeCol_ChildBg: (24, 23, 22, 255),
            dpg.mvThemeCol_PopupBg: (30, 28, 27, 255),
            dpg.mvThemeCol_MenuBarBg: (18, 17, 16, 255),

            dpg.mvThemeCol_Border: (70, 64, 54, 120),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (38, 36, 34, 255),
            dpg.mvThemeCol_FrameBgHovered: (52, 48, 44, 255),
            dpg.mvThemeCol_FrameBgActive: (66, 60, 54, 255),

            dpg.mvThemeCol_TitleBg: (14, 13, 12, 255),
            dpg.mvThemeCol_TitleBgActive: (26, 24, 22, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (14, 13, 12, 200),

            dpg.mvThemeCol_ScrollbarBg: (18, 17, 16, 255),
            dpg.mvThemeCol_ScrollbarGrab: (118, 92, 60, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (140, 110, 72, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (162, 128, 84, 255),

            dpg.mvThemeCol_Button: (118, 92, 60, 255),
            dpg.mvThemeCol_ButtonHovered: (140, 110, 72, 255),
            dpg.mvThemeCol_ButtonActive: (162, 128, 84, 255),

            dpg.mvThemeCol_Header: (54, 50, 46, 255),
            dpg.mvThemeCol_HeaderHovered: (66, 60, 54, 255),
            dpg.mvThemeCol_HeaderActive: (78, 70, 62, 255),

            dpg.mvThemeCol_Tab: (40, 38, 36, 255),
            dpg.mvThemeCol_TabHovered: (56, 52, 48, 255),
            dpg.mvThemeCol_TabActive: (48, 46, 44, 255),
            dpg.mvThemeCol_TabUnfocused: (32, 30, 28, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (40, 38, 36, 255),

            dpg.mvThemeCol_SliderGrab: (198, 156, 100, 255),
            dpg.mvThemeCol_SliderGrabActive: (222, 176, 116, 255),

            dpg.mvThemeCol_CheckMark: (230, 198, 140, 255),

            dpg.mvThemeCol_Text: (248, 240, 228, 255),
            dpg.mvThemeCol_TextDisabled: (155, 146, 134, 255),

            dpg.mvThemeCol_Separator: (70, 64, 54, 140),
            dpg.mvThemeCol_SeparatorHovered: (98, 86, 70, 255),
            dpg.mvThemeCol_SeparatorActive: (118, 102, 82, 255),

            dpg.mvThemeCol_ResizeGrip: (70, 64, 54, 60),
            dpg.mvThemeCol_ResizeGripHovered: (98, 86, 70, 200),
            dpg.mvThemeCol_ResizeGripActive: (118, 102, 82, 255),

            dpg.mvThemeCol_PlotHistogram: (170, 140, 96, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (196, 164, 112, 255),
        },
        "styles": {
            dpg.mvStyleVar_FrameRounding: 5,
            dpg.mvStyleVar_WindowRounding: 6,
            dpg.mvStyleVar_ChildRounding: 4,
            dpg.mvStyleVar_PopupRounding: 5,
            dpg.mvStyleVar_ScrollbarRounding: 6,
            dpg.mvStyleVar_GrabRounding: 3,
            dpg.mvStyleVar_TabRounding: 3,
            dpg.mvStyleVar_FramePadding: (9, 5),
            dpg.mvStyleVar_ItemSpacing: (9, 6),
            dpg.mvStyleVar_WindowPadding: (12, 12),
            dpg.mvStyleVar_FrameBorderSize: 0,
            dpg.mvStyleVar_WindowBorderSize: 0,
        },
    },

    "Slate Lichen": {
        "description": "Cool slate base with moss accents; calm but still organic",
        "colors": {
            dpg.mvThemeCol_WindowBg: (26, 28, 30, 255),
            dpg.mvThemeCol_ChildBg: (32, 34, 36, 255),
            dpg.mvThemeCol_PopupBg: (38, 40, 42, 255),
            dpg.mvThemeCol_MenuBarBg: (26, 28, 30, 255),

            dpg.mvThemeCol_Border: (58, 64, 68, 90),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (42, 46, 48, 255),
            dpg.mvThemeCol_FrameBgHovered: (52, 56, 58, 255),
            dpg.mvThemeCol_FrameBgActive: (62, 66, 68, 255),

            dpg.mvThemeCol_TitleBg: (22, 24, 26, 255),
            dpg.mvThemeCol_TitleBgActive: (32, 34, 36, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (22, 24, 26, 200),

            dpg.mvThemeCol_ScrollbarBg: (26, 28, 30, 255),
            dpg.mvThemeCol_ScrollbarGrab: (74, 92, 74, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (88, 110, 88, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (104, 128, 104, 255),

            dpg.mvThemeCol_Button: (74, 92, 74, 255),
            dpg.mvThemeCol_ButtonHovered: (88, 110, 88, 255),
            dpg.mvThemeCol_ButtonActive: (104, 128, 104, 255),

            dpg.mvThemeCol_Header: (50, 54, 56, 255),
            dpg.mvThemeCol_HeaderHovered: (62, 66, 68, 255),
            dpg.mvThemeCol_HeaderActive: (74, 78, 80, 255),

            dpg.mvThemeCol_Tab: (40, 42, 44, 255),
            dpg.mvThemeCol_TabHovered: (54, 56, 58, 255),
            dpg.mvThemeCol_TabActive: (48, 50, 52, 255),
            dpg.mvThemeCol_TabUnfocused: (34, 36, 38, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (42, 44, 46, 255),

            dpg.mvThemeCol_SliderGrab: (122, 146, 118, 255),
            dpg.mvThemeCol_SliderGrabActive: (144, 170, 140, 255),

            dpg.mvThemeCol_CheckMark: (172, 196, 164, 255),

            dpg.mvThemeCol_Text: (236, 236, 232, 255),
            dpg.mvThemeCol_TextDisabled: (138, 140, 138, 255),

            dpg.mvThemeCol_Separator: (58, 64, 68, 110),
            dpg.mvThemeCol_SeparatorHovered: (84, 92, 96, 255),
            dpg.mvThemeCol_SeparatorActive: (104, 112, 116, 255),

            dpg.mvThemeCol_ResizeGrip: (58, 64, 68, 50),
            dpg.mvThemeCol_ResizeGripHovered: (84, 92, 96, 190),
            dpg.mvThemeCol_ResizeGripActive: (104, 112, 116, 255),

            dpg.mvThemeCol_PlotHistogram: (132, 146, 134, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (156, 172, 156, 255),
        },
        "styles": {
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
        },
    },

    "Studio Dusk": {
        "description": "Dim neutral studio light: soft charcoal with muted terracotta cues",
        "colors": {
            dpg.mvThemeCol_WindowBg: (48, 46, 45, 255),
            dpg.mvThemeCol_ChildBg: (54, 52, 50, 255),
            dpg.mvThemeCol_PopupBg: (60, 58, 56, 255),
            dpg.mvThemeCol_MenuBarBg: (48, 46, 45, 255),

            dpg.mvThemeCol_Border: (70, 66, 62, 70),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (60, 58, 56, 255),
            dpg.mvThemeCol_FrameBgHovered: (68, 66, 64, 255),
            dpg.mvThemeCol_FrameBgActive: (76, 74, 72, 255),

            dpg.mvThemeCol_TitleBg: (42, 40, 39, 255),
            dpg.mvThemeCol_TitleBgActive: (52, 50, 48, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (42, 40, 39, 200),

            dpg.mvThemeCol_ScrollbarBg: (48, 46, 45, 255),
            dpg.mvThemeCol_ScrollbarGrab: (96, 82, 74, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (108, 92, 84, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (120, 104, 94, 255),

            dpg.mvThemeCol_Button: (96, 82, 74, 255),
            dpg.mvThemeCol_ButtonHovered: (108, 92, 84, 255),
            dpg.mvThemeCol_ButtonActive: (120, 104, 94, 255),

            dpg.mvThemeCol_Header: (54, 52, 50, 255),
            dpg.mvThemeCol_HeaderHovered: (64, 62, 60, 255),
            dpg.mvThemeCol_HeaderActive: (74, 70, 68, 255),

            dpg.mvThemeCol_Tab: (50, 48, 46, 255),
            dpg.mvThemeCol_TabHovered: (62, 60, 58, 255),
            dpg.mvThemeCol_TabActive: (56, 54, 52, 255),
            dpg.mvThemeCol_TabUnfocused: (44, 42, 41, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (52, 50, 48, 255),

            dpg.mvThemeCol_SliderGrab: (138, 124, 112, 255),
            dpg.mvThemeCol_SliderGrabActive: (156, 138, 126, 255),

            dpg.mvThemeCol_CheckMark: (178, 158, 142, 255),

            dpg.mvThemeCol_Text: (212, 206, 198, 255),
            dpg.mvThemeCol_TextDisabled: (142, 136, 128, 255),

            dpg.mvThemeCol_Separator: (70, 66, 62, 100),
            dpg.mvThemeCol_SeparatorHovered: (92, 84, 78, 255),
            dpg.mvThemeCol_SeparatorActive: (110, 102, 94, 255),

            dpg.mvThemeCol_ResizeGrip: (70, 66, 62, 45),
            dpg.mvThemeCol_ResizeGripHovered: (92, 84, 78, 180),
            dpg.mvThemeCol_ResizeGripActive: (110, 102, 94, 255),

            dpg.mvThemeCol_PlotHistogram: (142, 128, 116, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (166, 148, 134, 255),
        },
        "styles": {
            dpg.mvStyleVar_FrameRounding: 5,
            dpg.mvStyleVar_WindowRounding: 6,
            dpg.mvStyleVar_ChildRounding: 4,
            dpg.mvStyleVar_PopupRounding: 5,
            dpg.mvStyleVar_ScrollbarRounding: 6,
            dpg.mvStyleVar_GrabRounding: 3,
            dpg.mvStyleVar_TabRounding: 3,
            dpg.mvStyleVar_FramePadding: (8, 5),
            dpg.mvStyleVar_ItemSpacing: (9, 6),
            dpg.mvStyleVar_WindowPadding: (12, 12),
            dpg.mvStyleVar_FrameBorderSize: 0,
            dpg.mvStyleVar_WindowBorderSize: 0,
        },
    },

    "Dark Paper": {
        "description": "Soft black ink on dark paper; low-glare, warm parchment text",
        "colors": {
            dpg.mvThemeCol_WindowBg: (54, 52, 50, 255),
            dpg.mvThemeCol_ChildBg: (60, 58, 56, 255),
            dpg.mvThemeCol_PopupBg: (66, 64, 62, 255),
            dpg.mvThemeCol_MenuBarBg: (54, 52, 50, 255),

            dpg.mvThemeCol_Border: (74, 68, 62, 70),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (66, 64, 62, 255),
            dpg.mvThemeCol_FrameBgHovered: (74, 72, 70, 255),
            dpg.mvThemeCol_FrameBgActive: (82, 80, 78, 255),

            dpg.mvThemeCol_TitleBg: (46, 44, 42, 255),
            dpg.mvThemeCol_TitleBgActive: (58, 56, 54, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (46, 44, 42, 200),

            dpg.mvThemeCol_ScrollbarBg: (54, 52, 50, 255),
            dpg.mvThemeCol_ScrollbarGrab: (102, 90, 82, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (116, 102, 92, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (130, 114, 104, 255),

            dpg.mvThemeCol_Button: (102, 90, 82, 255),
            dpg.mvThemeCol_ButtonHovered: (116, 102, 92, 255),
            dpg.mvThemeCol_ButtonActive: (130, 114, 104, 255),

            dpg.mvThemeCol_Header: (58, 56, 54, 255),
            dpg.mvThemeCol_HeaderHovered: (68, 66, 64, 255),
            dpg.mvThemeCol_HeaderActive: (78, 76, 74, 255),

            dpg.mvThemeCol_Tab: (52, 50, 48, 255),
            dpg.mvThemeCol_TabHovered: (66, 64, 62, 255),
            dpg.mvThemeCol_TabActive: (58, 56, 54, 255),
            dpg.mvThemeCol_TabUnfocused: (46, 44, 42, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (54, 52, 50, 255),

            dpg.mvThemeCol_SliderGrab: (144, 130, 118, 255),
            dpg.mvThemeCol_SliderGrabActive: (162, 146, 132, 255),

            dpg.mvThemeCol_CheckMark: (188, 170, 150, 255),

            dpg.mvThemeCol_Text: (204, 196, 186, 255),
            dpg.mvThemeCol_TextDisabled: (140, 134, 126, 255),

            dpg.mvThemeCol_Separator: (74, 68, 62, 100),
            dpg.mvThemeCol_SeparatorHovered: (96, 88, 80, 255),
            dpg.mvThemeCol_SeparatorActive: (114, 104, 94, 255),

            dpg.mvThemeCol_ResizeGrip: (74, 68, 62, 45),
            dpg.mvThemeCol_ResizeGripHovered: (96, 88, 80, 180),
            dpg.mvThemeCol_ResizeGripActive: (114, 104, 94, 255),

            dpg.mvThemeCol_PlotHistogram: (144, 130, 112, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (168, 150, 130, 255),
        },
        "styles": {
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
        },
    },

    "Dusty Ochre": {
        "description": "Cool slate background with muted sand/ochre warmth; old paper feel",
        "colors": {
            dpg.mvThemeCol_WindowBg: (32, 34, 45, 255),
            dpg.mvThemeCol_ChildBg: (38, 40, 50, 255),
            dpg.mvThemeCol_PopupBg: (44, 46, 56, 255),
            dpg.mvThemeCol_MenuBarBg: (32, 34, 45, 255),

            dpg.mvThemeCol_Border: (62, 64, 74, 80),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (46, 48, 58, 255),
            dpg.mvThemeCol_FrameBgHovered: (54, 56, 66, 255),
            dpg.mvThemeCol_FrameBgActive: (62, 64, 74, 255),

            dpg.mvThemeCol_TitleBg: (26, 28, 38, 255),
            dpg.mvThemeCol_TitleBgActive: (36, 38, 48, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (26, 28, 38, 200),

            dpg.mvThemeCol_ScrollbarBg: (32, 34, 45, 255),
            dpg.mvThemeCol_ScrollbarGrab: (92, 86, 76, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (104, 96, 84, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (116, 106, 92, 255),

            dpg.mvThemeCol_Button: (78, 74, 68, 255),
            dpg.mvThemeCol_ButtonHovered: (90, 84, 76, 255),
            dpg.mvThemeCol_ButtonActive: (100, 94, 84, 255),

            dpg.mvThemeCol_Header: (52, 54, 64, 255),
            dpg.mvThemeCol_HeaderHovered: (62, 64, 74, 255),
            dpg.mvThemeCol_HeaderActive: (72, 74, 84, 255),

            dpg.mvThemeCol_Tab: (46, 48, 58, 255),
            dpg.mvThemeCol_TabHovered: (60, 62, 72, 255),
            dpg.mvThemeCol_TabActive: (54, 56, 66, 255),
            dpg.mvThemeCol_TabUnfocused: (42, 44, 54, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (50, 52, 62, 255),

            dpg.mvThemeCol_SliderGrab: (130, 118, 100, 255),
            dpg.mvThemeCol_SliderGrabActive: (148, 134, 114, 255),

            dpg.mvThemeCol_CheckMark: (178, 162, 140, 255),

            dpg.mvThemeCol_Text: (226, 220, 212, 255),
            dpg.mvThemeCol_TextDisabled: (142, 138, 132, 255),

            dpg.mvThemeCol_Separator: (62, 64, 74, 100),
            dpg.mvThemeCol_SeparatorHovered: (88, 90, 100, 255),
            dpg.mvThemeCol_SeparatorActive: (106, 108, 118, 255),

            dpg.mvThemeCol_ResizeGrip: (62, 64, 74, 45),
            dpg.mvThemeCol_ResizeGripHovered: (88, 90, 100, 180),
            dpg.mvThemeCol_ResizeGripActive: (106, 108, 118, 255),

            dpg.mvThemeCol_PlotHistogram: (144, 128, 106, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (168, 146, 122, 255),
        },
        "styles": {
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
        },
    },

    "Moonlit Paper": {
        "description": "Background at (32,34,45) with soft parchment text and cedar ink",
        "colors": {
            dpg.mvThemeCol_WindowBg: (32, 34, 45, 255),
            dpg.mvThemeCol_ChildBg: (38, 40, 50, 255),
            dpg.mvThemeCol_PopupBg: (44, 46, 56, 255),
            dpg.mvThemeCol_MenuBarBg: (32, 34, 45, 255),

            dpg.mvThemeCol_Border: (62, 64, 74, 80),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (46, 48, 58, 255),
            dpg.mvThemeCol_FrameBgHovered: (54, 56, 66, 255),
            dpg.mvThemeCol_FrameBgActive: (62, 64, 74, 255),

            dpg.mvThemeCol_TitleBg: (26, 28, 38, 255),
            dpg.mvThemeCol_TitleBgActive: (36, 38, 48, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (26, 28, 38, 200),

            dpg.mvThemeCol_ScrollbarBg: (32, 34, 45, 255),
            dpg.mvThemeCol_ScrollbarGrab: (82, 84, 92, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (92, 94, 102, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (102, 104, 112, 255),

            dpg.mvThemeCol_Button: (82, 84, 92, 255),
            dpg.mvThemeCol_ButtonHovered: (92, 94, 102, 255),
            dpg.mvThemeCol_ButtonActive: (102, 104, 112, 255),

            dpg.mvThemeCol_Header: (52, 54, 64, 255),
            dpg.mvThemeCol_HeaderHovered: (62, 64, 74, 255),
            dpg.mvThemeCol_HeaderActive: (72, 74, 84, 255),

            dpg.mvThemeCol_Tab: (46, 48, 58, 255),
            dpg.mvThemeCol_TabHovered: (60, 62, 72, 255),
            dpg.mvThemeCol_TabActive: (54, 56, 66, 255),
            dpg.mvThemeCol_TabUnfocused: (42, 44, 54, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (50, 52, 62, 255),

            dpg.mvThemeCol_SliderGrab: (128, 120, 112, 255),
            dpg.mvThemeCol_SliderGrabActive: (146, 136, 126, 255),

            dpg.mvThemeCol_CheckMark: (180, 170, 160, 255),

            dpg.mvThemeCol_Text: (226, 218, 208, 255),
            dpg.mvThemeCol_TextDisabled: (142, 136, 130, 255),

            dpg.mvThemeCol_Separator: (62, 64, 74, 100),
            dpg.mvThemeCol_SeparatorHovered: (88, 90, 100, 255),
            dpg.mvThemeCol_SeparatorActive: (106, 108, 118, 255),

            dpg.mvThemeCol_ResizeGrip: (62, 64, 74, 45),
            dpg.mvThemeCol_ResizeGripHovered: (88, 90, 100, 180),
            dpg.mvThemeCol_ResizeGripActive: (106, 108, 118, 255),

            dpg.mvThemeCol_PlotHistogram: (144, 128, 106, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (168, 146, 122, 255),
        },
        "styles": {
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
        },
    },

    "Graphite Moss": {
        "description": "Background at (32,34,45) with moss accents; cool slate, organic feel",
        "colors": {
            dpg.mvThemeCol_WindowBg: (32, 34, 45, 255),
            dpg.mvThemeCol_ChildBg: (38, 40, 50, 255),
            dpg.mvThemeCol_PopupBg: (44, 46, 56, 255),
            dpg.mvThemeCol_MenuBarBg: (32, 34, 45, 255),

            dpg.mvThemeCol_Border: (62, 68, 72, 80),
            dpg.mvThemeCol_BorderShadow: (0, 0, 0, 0),

            dpg.mvThemeCol_FrameBg: (46, 50, 58, 255),
            dpg.mvThemeCol_FrameBgHovered: (54, 58, 66, 255),
            dpg.mvThemeCol_FrameBgActive: (62, 66, 74, 255),

            dpg.mvThemeCol_TitleBg: (26, 28, 38, 255),
            dpg.mvThemeCol_TitleBgActive: (36, 38, 48, 255),
            dpg.mvThemeCol_TitleBgCollapsed: (26, 28, 38, 200),

            dpg.mvThemeCol_ScrollbarBg: (32, 34, 45, 255),
            dpg.mvThemeCol_ScrollbarGrab: (72, 88, 78, 255),
            dpg.mvThemeCol_ScrollbarGrabHovered: (84, 100, 90, 255),
            dpg.mvThemeCol_ScrollbarGrabActive: (96, 112, 102, 255),

            dpg.mvThemeCol_Button: (72, 88, 78, 255),
            dpg.mvThemeCol_ButtonHovered: (84, 100, 90, 255),
            dpg.mvThemeCol_ButtonActive: (96, 112, 102, 255),

            dpg.mvThemeCol_Header: (52, 56, 64, 255),
            dpg.mvThemeCol_HeaderHovered: (62, 66, 74, 255),
            dpg.mvThemeCol_HeaderActive: (72, 76, 84, 255),

            dpg.mvThemeCol_Tab: (46, 50, 58, 255),
            dpg.mvThemeCol_TabHovered: (60, 64, 72, 255),
            dpg.mvThemeCol_TabActive: (54, 58, 66, 255),
            dpg.mvThemeCol_TabUnfocused: (42, 46, 54, 255),
            dpg.mvThemeCol_TabUnfocusedActive: (50, 54, 62, 255),

            dpg.mvThemeCol_SliderGrab: (116, 132, 120, 255),
            dpg.mvThemeCol_SliderGrabActive: (134, 150, 138, 255),

            dpg.mvThemeCol_CheckMark: (164, 184, 168, 255),

            dpg.mvThemeCol_Text: (222, 222, 218, 255),
            dpg.mvThemeCol_TextDisabled: (140, 142, 138, 255),

            dpg.mvThemeCol_Separator: (62, 68, 72, 100),
            dpg.mvThemeCol_SeparatorHovered: (88, 96, 100, 255),
            dpg.mvThemeCol_SeparatorActive: (108, 116, 120, 255),

            dpg.mvThemeCol_ResizeGrip: (62, 68, 72, 45),
            dpg.mvThemeCol_ResizeGripHovered: (88, 96, 100, 180),
            dpg.mvThemeCol_ResizeGripActive: (108, 116, 120, 255),

            dpg.mvThemeCol_PlotHistogram: (130, 144, 128, 255),
            dpg.mvThemeCol_PlotHistogramHovered: (154, 170, 150, 255),
        },
        "styles": {
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

    dpg.create_context()

    # Setup font (must be before any widgets)
    setup_font("DejaVu Sans", size=14)

    # Create all themes
    theme_ids = {name: create_theme(name) for name in THEMES}

    # Create viewport
    dpg.create_viewport(title="Elliptica Theme Preview", width=600, height=800)

    with dpg.window(label="Theme Preview", tag="primary_window"):
        # Theme selection
        dpg.add_text("Theme Selection", color=(180, 175, 165))
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
    main()
