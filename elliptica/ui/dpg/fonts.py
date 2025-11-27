"""Font configuration for DearPyGui UI.

DearPyGui's default font (ProggyClean) only supports basic Latin characters.
This module loads DejaVu Sans from matplotlib's bundled fonts, which has
excellent Unicode coverage including mathematical symbols.

Note on font crispness:
DearPyGui disables Retina framebuffer on macOS (GLFW_COCOA_RETINA_FRAMEBUFFER=FALSE),
meaning the entire UI is rendered at logical pixels then scaled up by the OS.
To compensate, we detect the display scale factor and rasterize fonts at 2x size,
then use set_global_font_scale(0.5) so they render at the intended visual size
but with 2x the glyph detail.
"""

from pathlib import Path
import sys
import dearpygui.dearpygui as dpg


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
        # Look for "Retina" in output
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


# Unicode code points for characters we need
MATH_SYMBOLS = [
    0x2202,  # ∂ PARTIAL DIFFERENTIAL
    0x2207,  # ∇ NABLA
    0x03C6,  # φ GREEK SMALL LETTER PHI
    0x2099,  # ₙ LATIN SUBSCRIPT SMALL LETTER N
    0x00B2,  # ² SUPERSCRIPT TWO
    0x00D7,  # × MULTIPLICATION SIGN (used in resolution labels)
]

# Greek letters range (for potential future use)
GREEK_RANGE = (0x0370, 0x03FF)

# Mathematical operators range
MATH_OPERATORS_RANGE = (0x2200, 0x22FF)

# Subscript/superscript range
SUB_SUPER_RANGE = (0x2070, 0x209F)


def setup_fonts(size: int = 14) -> int | None:
    """Load and configure fonts with math symbol support.

    Must be called after dpg.create_context() and before any widgets are created.

    On Retina displays, fonts are rasterized at 2x size and globally scaled down
    to achieve crisp rendering despite DearPyGui's disabled Retina framebuffer.

    Args:
        size: Desired visual font size in logical pixels

    Returns:
        Font ID if successful, None if font not found
    """
    font_path = _find_dejavu_font()
    if font_path is None:
        print("Warning: DejaVu Sans not found, using default font (math symbols may not render)")
        return None

    # Detect display scale and rasterize at higher resolution for crispness
    display_scale = _get_display_scale()
    raster_size = int(size * display_scale)

    with dpg.font_registry():
        with dpg.font(str(font_path), raster_size) as font_id:
            # Add default Latin range (this is automatic but explicit is clearer)
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)

            # Add Greek letters range (includes φ)
            dpg.add_font_range(GREEK_RANGE[0], GREEK_RANGE[1])

            # Add mathematical operators range (includes ∂, ∇)
            dpg.add_font_range(MATH_OPERATORS_RANGE[0], MATH_OPERATORS_RANGE[1])

            # Add subscript/superscript range (includes ₙ, ²)
            dpg.add_font_range(SUB_SUPER_RANGE[0], SUB_SUPER_RANGE[1])

            # Explicitly add specific characters we know we need
            dpg.add_font_chars(MATH_SYMBOLS)

    # Bind as global default font
    dpg.bind_font(font_id)

    # Scale down to intended visual size (glyphs have 2x detail on Retina)
    if display_scale > 1.0:
        dpg.set_global_font_scale(1.0 / display_scale)

    return font_id
