"""Font configuration for DearPyGui UI.

DearPyGui's default font (ProggyClean) only supports basic Latin characters.
This module loads DejaVu Sans from matplotlib's bundled fonts, which has
excellent Unicode coverage including mathematical symbols.
"""

from pathlib import Path
import dearpygui.dearpygui as dpg


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

    Args:
        size: Font size in pixels

    Returns:
        Font ID if successful, None if font not found
    """
    font_path = _find_dejavu_font()
    if font_path is None:
        print("Warning: DejaVu Sans not found, using default font (math symbols may not render)")
        return None

    with dpg.font_registry():
        with dpg.font(str(font_path), size) as font_id:
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

    return font_id
