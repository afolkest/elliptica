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
        L="clipnorm(lic, 0.5, 99.5)",
        C="0",
        H="0",
        description="Simple grayscale from LIC texture",
    ),
    "Ink Wash": ExpressionPreset(
        name="Ink Wash",
        L="0.15 + 0.7 * clipnorm(lic, 0.5, 99.5)",
        C="0.02",
        H="220",
        description="Subtle blue-gray ink wash effect",
    ),
    "Warm Copper": ExpressionPreset(
        name="Warm Copper",
        L="0.25 + 0.5 * clipnorm(lic, 0.5, 99.5)",
        C="0.08",
        H="35",
        description="Warm copper/bronze tones",
    ),
    "Cool Blue": ExpressionPreset(
        name="Cool Blue",
        L="0.2 + 0.6 * clipnorm(lic, 0.5, 99.5)",
        C="0.1",
        H="240",
        description="Cool blue tones",
    ),
    "Field Intensity": ExpressionPreset(
        name="Field Intensity",
        L="0.2 + 0.5 * clipnorm(lic, 0.5, 99.5)",
        C="0.15 * clipnorm(mag, 1, 99)",
        H="200",
        description="Chroma from field magnitude (requires mag)",
    ),
    "Hue from Angle": ExpressionPreset(
        name="Hue from Angle",
        L="0.3 + 0.4 * clipnorm(lic, 0.5, 99.5)",
        C="0.1",
        H="180 + 180 * atan2(ey, ex) / 3.14159",
        description="Hue encodes field direction (requires ex, ey)",
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
    ("lic", "LIC texture (grayscale field visualization)"),
    ("mag", "Field magnitude sqrt(ex² + ey²)"),
    ("ex", "Field X component"),
    ("ey", "Field Y component"),
]

AVAILABLE_FUNCTIONS = [
    ("clipnorm(x, lo, hi)", "Normalize with percentile clipping (most common)"),
    ("normalize(x)", "Normalize to [0,1] using min/max"),
    ("smoothstep(e0, e1, x)", "Smooth interpolation between edges"),
    ("lerp(a, b, t)", "Linear interpolation: a + t*(b-a)"),
    ("clamp(x, lo, hi)", "Clamp value to range"),
    ("abs(x)", "Absolute value"),
    ("sqrt(x)", "Square root"),
    ("sin(x), cos(x)", "Trigonometric functions"),
    ("atan2(y, x)", "Two-argument arctangent (radians)"),
    ("log(x), exp(x)", "Natural log and exponential"),
    ("pow(x, y)", "Power function x^y"),
]
