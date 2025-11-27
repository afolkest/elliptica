"""Built-in expression presets for ColorConfig.

Presets are named configurations of L/C/H expressions that users can
select as starting points and customize.
"""

from dataclasses import dataclass


@dataclass
class ExpressionPreset:
    """A named preset of L/C/H expressions."""
    name: str
    L: str
    C: str
    H: str
    description: str = ""


# Built-in presets
BUILTIN_PRESETS: dict[str, ExpressionPreset] = {
    "Grayscale": ExpressionPreset(
        name="Grayscale",
        L="clipnorm(lic, 1.5, 98.5)",
        C="0",
        H="0",
        description="Simple grayscale from LIC texture",
    ),
    "Ink Wash": ExpressionPreset(
        name="Ink Wash",
        L="clipnorm(lic, 1.5, 98.5)",
        C="0.02",
        H="220",
        description="Subtle blue-gray ink wash effect",
    ),
    "Warm Copper": ExpressionPreset(
        name="Warm Copper",
        L="clipnorm(lic, 1.5, 98.5)",
        C="0.08",
        H="35",
        description="Warm copper/bronze tones",
    ),
    "Cool Blue": ExpressionPreset(
        name="Cool Blue",
        L="clipnorm(lic, 1.5, 98.5)",
        C="0.1",
        H="240",
        description="Cool blue tones",
    ),
    "Field Intensity": ExpressionPreset(
        name="Field Intensity",
        L="clipnorm(lic, 1.5, 98.5)",
        C="0.15 * clipnorm(mag, 1, 99)",
        H="200",
        description="Chroma from field magnitude",
    ),
    "Hue from Angle": ExpressionPreset(
        name="Hue from Angle",
        L="clipnorm(lic, 1.5, 98.5)",
        C="0.1",
        H="180 + 180 * atan2(ey, ex) / pi",
        description="Hue encodes field direction",
    ),
}


def get_preset(name: str) -> ExpressionPreset | None:
    """Get a preset by name."""
    return BUILTIN_PRESETS.get(name)


def list_presets() -> list[str]:
    """List all available preset names."""
    return list(BUILTIN_PRESETS.keys())


# Reference information for UI
AVAILABLE_VARIABLES = [
    ("lic", "LIC texture intensity"),
    ("mag", "Field magnitude √(ex² + ey²)"),
    ("ex", "Field X component"),
    ("ey", "Field Y component"),
]

# Functions are grouped by type:
# - Global: uses statistics from entire image (percentiles, min/max)
# - Per-pixel: operates on each pixel independently
AVAILABLE_FUNCTIONS = [
    # Global normalization (uses whole-image statistics)
    ("clipnorm(x, lo, hi)", "Global: normalize using percentiles (e.g. clipnorm(lic, 1, 99))"),
    ("normalize(x)", "Global: normalize to [0,1] using min/max"),
    # Per-pixel transformations
    ("smoothstep(lo, hi, x)", "S-curve: 0 when x≤lo, 1 when x≥hi, smooth between"),
    ("lerp(a, b, t)", "Linear blend: a*(1-t) + b*t"),
    ("clamp(x, lo, hi)", "Constrain x to [lo, hi]"),
    ("abs(x)", "Absolute value"),
    ("sqrt(x), pow(x, y)", "Square root, power"),
    ("sin(x), cos(x)", "Trig functions (radians)"),
    ("atan2(y, x)", "Angle of vector (x,y) → [-π, π] radians"),
    ("log(x), exp(x)", "Natural log, exponential"),
]
