import copy
import numpy as np
from collections import OrderedDict
from pathlib import Path
from elliptica.colorspace.oklch_palette import build_oklch_lut

COLOR_PALETTES: dict[str, np.ndarray] = {
    "Ink & Gold": np.array(
        [
            (0.02, 0.04, 0.10),
            (0.10, 0.18, 0.28),
            (0.28, 0.42, 0.40),
            (0.48, 0.56, 0.40),
            (0.75, 0.66, 0.30),
            (0.95, 0.80, 0.22),
            (1.00, 0.90, 0.60),
        ],
        dtype=np.float32,
    ),
    "Deep Ocean": np.array(
        [
            (0.05, 0.05, 0.20),
            (0.10, 0.15, 0.40),
            (0.20, 0.32, 0.62),
            (0.28, 0.50, 0.78),
            (0.45, 0.70, 0.88),
            (0.68, 0.82, 0.94),
            (0.90, 0.96, 1.00),
        ],
        dtype=np.float32,
    ),
    "Ember Ash": np.array(
        [
            (0.00, 0.00, 0.00),
            (0.10, 0.05, 0.02),
            (0.30, 0.12, 0.04),
            (0.60, 0.26, 0.08),
            (0.75, 0.55, 0.35),
            (0.88, 0.89, 0.90),
        ],
        dtype=np.float32,
    ),
    "Twilight Peach": np.array(
        [
            (0.15, 0.05, 0.30),
            (0.40, 0.20, 0.55),
            (0.65, 0.45, 0.75),
            (0.90, 0.75, 0.85),
            (1.00, 0.95, 0.90),
            (0.95, 0.80, 0.65),
            (0.85, 0.60, 0.45),
            (0.65, 0.40, 0.35),
            (0.35, 0.20, 0.30),
        ],
        dtype=np.float32,
    ),
    "Twilight Magenta": np.array(
        [
            (0.18, 0.07, 0.22),  # Dark purple
            (0.26, 0.11, 0.35),  # Deep purple
            (0.33, 0.17, 0.48),  # Purple
            (0.40, 0.30, 0.60),  # Medium purple
            (0.50, 0.45, 0.70),  # Light purple
            (0.65, 0.60, 0.80),  # Lavender
            (0.80, 0.75, 0.88),  # Very light lavender
            (0.92, 0.88, 0.94),  # Near white
            (0.95, 0.90, 0.85),  # Peachy white
            (0.88, 0.70, 0.55),  # Peach/gold
            (0.75, 0.50, 0.40),  # Orange-brown
            (0.50, 0.25, 0.35),  # Purple-brown
            (0.35, 0.15, 0.30),  # Dark purple-red
            (0.18, 0.08, 0.21),  # Dark purple (wrap)
        ],
        dtype=np.float32,
    ),
    "Fire & Ice": np.array(
        [
            (0.10, 0.10, 0.50),
            (0.20, 0.20, 0.70),
            (0.40, 0.40, 0.90),
            (0.70, 0.70, 1.00),
            (1.00, 1.00, 1.00),
            (1.00, 0.90, 0.70),
            (1.00, 0.70, 0.40),
            (0.90, 0.40, 0.20),
            (0.70, 0.20, 0.10),
        ],
        dtype=np.float32,
    ),
    "Stark B&W": np.array(
        [
            (0.00, 0.00, 0.00),  # pure black
            (0.00, 0.00, 0.00),  # black
            (0.15, 0.15, 0.15),  # dark gray
            (1.00, 1.00, 1.00),  # white
            (1.00, 1.00, 1.00),  # white
            (0.15, 0.15, 0.15),  # dark gray
            (0.00, 0.00, 0.00),  # black
        ],
        dtype=np.float32,
    ),
    "Ink Wash": np.array(
        [
            (0.00, 0.00, 0.00),  # pure black
            (0.00, 0.00, 0.00),  # pure black
            (0.03, 0.03, 0.03),  # very dark
            (0.08, 0.08, 0.08),  # dark
            (0.90, 0.90, 0.90),  # light
            (0.96, 0.96, 0.96),  # near white
            (1.00, 1.00, 1.00),  # pure white
        ],
        dtype=np.float32,
    ),
    "Inky Washy": np.array(
        [
            (0.00, 0.00, 0.00),  # pure black
            (0.00, 0.00, 0.00),  # pure black
            (0.03, 0.03, 0.03),  # very dark
            (0.08, 0.08, 0.08),  # dark
            (0.99, 0.99, 0.99),  # light
            (0.00, 0.00, 0.00),  # near white
            (0.00, 0.00, 0.00),  # pure white
        ],
        dtype=np.float32,
    ),
    "PuOr": np.array(
        [
            (0.498, 0.231, 0.031),  # Dark orange
            (0.702, 0.345, 0.024),  # Orange
            (0.878, 0.510, 0.078),  # Light orange
            (0.992, 0.722, 0.388),  # Pale orange
            (0.969, 0.969, 0.969),  # White
            (0.847, 0.855, 0.922),  # Pale purple
            (0.698, 0.671, 0.824),  # Light purple
            (0.502, 0.451, 0.675),  # Purple
            (0.329, 0.153, 0.533),  # Dark purple
        ],
        dtype=np.float32,
    ),
    "Coolwarm": np.array(
        [
            (0.231, 0.298, 0.753),  # Cool blue
            (0.404, 0.475, 0.859),  # Light blue
            (0.608, 0.667, 0.925),  # Pale blue
            (0.780, 0.839, 0.965),  # Very pale blue
            (0.933, 0.933, 0.933),  # White/gray
            (0.969, 0.792, 0.729),  # Pale red
            (0.933, 0.573, 0.490),  # Light red
            (0.843, 0.329, 0.333),  # Red
            (0.706, 0.016, 0.149),  # Dark red
        ],
        dtype=np.float32,
    ),
    "Aurora": np.array(
        [
            (0.05, 0.14, 0.14),
            (0.10, 0.24, 0.24),
            (0.22, 0.40, 0.40),
            (0.36, 0.52, 0.58),
            (0.50, 0.56, 0.72),
            (0.64, 0.52, 0.80),
            (0.76, 0.60, 0.86),
            (0.86, 0.72, 0.90),
            (0.94, 0.86, 0.96),
        ],
        dtype=np.float32,
    ),
    "Bronze Teal": np.array(
        [
            (0.08, 0.05, 0.02),
            (0.16, 0.10, 0.05),
            (0.32, 0.20, 0.10),
            (0.56, 0.36, 0.14),
            (0.78, 0.56, 0.22),
            (0.56, 0.66, 0.54),
            (0.36, 0.68, 0.64),
            (0.22, 0.58, 0.58),
            (0.14, 0.44, 0.46),
        ],
        dtype=np.float32,
    ),
    "Electric": np.array(
        [
            (0.00, 0.00, 0.00),
            (0.10, 0.00, 0.20),
            (0.20, 0.00, 0.40),
            (0.40, 0.10, 0.60),
            (0.60, 0.30, 0.80),
            (0.80, 0.60, 0.90),
            (0.90, 0.80, 0.95),
            (0.95, 0.90, 1.00),
            (1.00, 1.00, 1.00),
        ],
        dtype=np.float32,
    ),
    "Jade Lavender Gold": np.array(
        [
            (0.02, 0.22, 0.18),
            (0.08, 0.44, 0.34),
            (0.26, 0.66, 0.52),
            (0.86, 0.80, 0.96),
            (0.76, 0.66, 0.90),
            (0.96, 0.84, 0.46),
            (1.00, 0.90, 0.62),
        ],
        dtype=np.float32,
    ),
    "Seafoam Sunset Plum": np.array(
        [
            (0.06, 0.24, 0.22),
            (0.22, 0.60, 0.58),
            (0.58, 0.90, 0.86),
            (1.00, 0.70, 0.50),
            (0.98, 0.54, 0.38),
            (0.70, 0.52, 0.78),
            (0.52, 0.36, 0.64),
        ],
        dtype=np.float32,
    ),
    "Viridian Ochre Cerulean": np.array(
        [
            (0.02, 0.16, 0.12),
            (0.06, 0.32, 0.26),
            (0.10, 0.50, 0.40),
            (0.86, 0.72, 0.26),
            (0.98, 0.84, 0.42),
            (0.20, 0.48, 0.74),
            (0.10, 0.34, 0.60),
        ],
        dtype=np.float32,
    ),
}

DEFAULT_COLOR_PALETTE_NAME = "Ink Wash"
PALETTE_SCHEMA_VERSION = 2
OKLCH_DEFAULT_RELATIVE_CHROMA = True
OKLCH_DEFAULT_INTERP_MIX = 1.0
OKLCH_COLORMAP_LUT_SIZE = 16
_PALETTE_LUT_CACHE_MAX = 32
_PALETTE_LUT_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def _rgb_colors_to_stops(colors: np.ndarray) -> list[dict]:
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size == 0:
        return []
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("RGB colors must have shape (N, 3)")

    count = colors.shape[0]
    if count == 1:
        positions = np.array([0.0], dtype=np.float32)
    else:
        positions = np.linspace(0.0, 1.0, count, dtype=np.float32)

    stops = []
    for pos, color in zip(positions, colors):
        stops.append({
            "pos": float(pos),
            "r": float(color[0]),
            "g": float(color[1]),
            "b": float(color[2]),
        })
    return stops


def _rgb_stops_to_colors(stops: list[dict]) -> np.ndarray:
    if not stops:
        return np.zeros((0, 3), dtype=np.float32)
    stops_sorted = sorted(stops, key=lambda s: float(s["pos"]))
    return np.array(
        [[float(s["r"]), float(s["g"]), float(s["b"])] for s in stops_sorted],
        dtype=np.float32,
    )


def _build_rgb_lut_from_stops(stops: list[dict], size: int = 256) -> np.ndarray:
    if size <= 0:
        raise ValueError("LUT size must be positive")
    if not stops:
        return np.zeros((size, 3), dtype=np.float32)

    stops_sorted = sorted(stops, key=lambda s: float(s["pos"]))
    positions = np.array([float(s["pos"]) for s in stops_sorted], dtype=np.float32)
    colors = np.array(
        [[float(s["r"]), float(s["g"]), float(s["b"])] for s in stops_sorted],
        dtype=np.float32,
    )
    positions = np.clip(positions, 0.0, 1.0)

    if len(stops_sorted) == 1:
        return np.tile(colors[0], (size, 1))

    samples = np.linspace(0.0, 1.0, size, dtype=np.float32)
    lut = np.empty((size, 3), dtype=np.float32)
    for channel in range(3):
        lut[:, channel] = np.interp(samples, positions, colors[:, channel])
    return lut


def _palette_spec_is_deleted(spec: dict | None) -> bool:
    if spec is None or not isinstance(spec, dict):
        return True
    return bool(spec.get("deleted", False))


def _palette_spec_cache_key(spec: dict, size: int) -> tuple:
    space = spec.get("space", "rgb")
    stops = spec.get("stops", [])
    if space == "oklch":
        relative_chroma = bool(spec.get("relative_chroma", OKLCH_DEFAULT_RELATIVE_CHROMA))
        interp_mix = float(spec.get("interp_mix", OKLCH_DEFAULT_INTERP_MIX))
        stop_key = tuple(sorted(
            (
                float(s["pos"]),
                float(s["L"]),
                float(s["C"]),
                float(s["H"]),
            )
            for s in stops
        ))
        return ("oklch", size, relative_chroma, interp_mix, stop_key)

    stop_key = tuple(sorted(
        (
            float(s["pos"]),
            float(s["r"]),
            float(s["g"]),
            float(s["b"]),
        )
        for s in stops
    ))
    return ("rgb", size, stop_key)


def _get_cached_palette_lut(cache_key: tuple) -> np.ndarray | None:
    lut = _PALETTE_LUT_CACHE.get(cache_key)
    if lut is not None:
        _PALETTE_LUT_CACHE.move_to_end(cache_key)
    return lut


def _set_cached_palette_lut(cache_key: tuple, lut: np.ndarray) -> None:
    _PALETTE_LUT_CACHE[cache_key] = lut
    _PALETTE_LUT_CACHE.move_to_end(cache_key)
    if len(_PALETTE_LUT_CACHE) > _PALETTE_LUT_CACHE_MAX:
        _PALETTE_LUT_CACHE.popitem(last=False)


def _palette_spec_to_lut(spec: dict, size: int = 256) -> np.ndarray:
    cache_key = _palette_spec_cache_key(spec, size)
    cached = _get_cached_palette_lut(cache_key)
    if cached is not None:
        return cached

    space = spec.get("space", "rgb")
    if space == "oklch":
        lut = build_oklch_lut(
            spec.get("stops", []),
            size=size,
            relative_chroma=bool(spec.get("relative_chroma", OKLCH_DEFAULT_RELATIVE_CHROMA)),
            interp_mix=float(spec.get("interp_mix", OKLCH_DEFAULT_INTERP_MIX)),
        )
    else:
        lut = _build_rgb_lut_from_stops(spec.get("stops", []), size=size)

    _set_cached_palette_lut(cache_key, lut)
    return lut


def _palette_spec_to_colormap_colors(spec: dict) -> np.ndarray:
    space = spec.get("space", "rgb")
    if space == "rgb":
        return _rgb_stops_to_colors(spec.get("stops", []))
    return _palette_spec_to_lut(spec, size=OKLCH_COLORMAP_LUT_SIZE)


def _rgb_palette_spec_from_colors(colors: np.ndarray) -> dict:
    return {
        "space": "rgb",
        "stops": _rgb_colors_to_stops(colors),
    }


# User palette persistence
USER_DATA_DIR = Path.home() / ".elliptica"
USER_PALETTES_PATH = USER_DATA_DIR / "palettes.json"


def _write_palette_backup() -> None:
    bak_path = USER_PALETTES_PATH.with_suffix(USER_PALETTES_PATH.suffix + ".bak")
    if bak_path.exists():
        return
    try:
        import shutil
        shutil.copyfile(USER_PALETTES_PATH, bak_path)
    except FileNotFoundError:
        pass


def _load_user_palette_specs() -> dict[str, dict]:
    """Load user palette specs from JSON (additions/overrides)."""
    if not USER_PALETTES_PATH.exists():
        return {}

    import json
    try:
        with open(USER_PALETTES_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Corrupt or unreadable file - log and return empty
        print(f"Warning: Could not load user palettes from {USER_PALETTES_PATH}: {e}")
        return {}

    if isinstance(data, dict) and "palettes" in data:
        palettes = data.get("palettes", {})
        return palettes if isinstance(palettes, dict) else {}

    if not isinstance(data, dict):
        return {}

    # Legacy format: {name: [[r,g,b], ...]}
    migrated: dict[str, dict] = {}
    for name, colors in data.items():
        if colors is None:
            migrated[name] = {"deleted": True}
            continue
        arr = np.asarray(colors, dtype=np.float32)
        if arr.size == 0:
            migrated[name] = {"deleted": True}
            continue
        migrated[name] = _rgb_palette_spec_from_colors(arr)

    _write_palette_backup()
    _save_user_palette_specs(migrated)
    return migrated


def _save_user_palette_specs(palettes: dict[str, dict]):
    """Save user palette specs to JSON."""
    import json
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "version": PALETTE_SCHEMA_VERSION,
        "palettes": palettes,
    }
    with open(USER_PALETTES_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def _build_default_palette_specs() -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for name, colors in COLOR_PALETTES.items():
        specs[name] = _rgb_palette_spec_from_colors(colors)
    return specs


def _build_runtime_palette_specs() -> dict[str, dict]:
    """Build runtime palette specs: defaults + user overrides - user deletions."""
    user_palettes = _load_user_palette_specs()
    merged = _build_default_palette_specs()

    for name, spec in user_palettes.items():
        if _palette_spec_is_deleted(spec):
            merged.pop(name, None)
        else:
            merged[name] = spec
    return merged


def _build_runtime_palettes(palette_specs: dict[str, dict]) -> dict[str, np.ndarray]:
    palettes: dict[str, np.ndarray] = {}
    for name, spec in palette_specs.items():
        if _palette_spec_is_deleted(spec):
            continue
        colors = _palette_spec_to_colormap_colors(spec)
        if colors.size == 0:
            continue
        palettes[name] = colors
    return palettes


def _build_palette_luts(palette_specs: dict[str, dict]) -> dict[str, np.ndarray]:
    luts: dict[str, np.ndarray] = {}
    for name, spec in palette_specs.items():
        if _palette_spec_is_deleted(spec):
            continue
        luts[name] = _palette_spec_to_lut(spec)
    return luts


_RUNTIME_PALETTE_SPECS = _build_runtime_palette_specs()
_RUNTIME_PALETTES = _build_runtime_palettes(_RUNTIME_PALETTE_SPECS)
PALETTE_LUTS: dict[str, np.ndarray] = _build_palette_luts(_RUNTIME_PALETTE_SPECS)


def list_palette_colormap_colors() -> dict[str, np.ndarray]:
    """Return palette colors used for DPG colormaps."""
    return dict(_RUNTIME_PALETTES)


def get_palette_spec(name: str) -> dict | None:
    """Return a copy of the palette spec for a palette name."""
    spec = _RUNTIME_PALETTE_SPECS.get(name)
    if spec is None:
        return None
    return copy.deepcopy(spec)


def list_color_palettes() -> tuple[str, ...]:
    return tuple(PALETTE_LUTS.keys())


def _get_palette_lut(name: str | None) -> np.ndarray:
    if name and name in PALETTE_LUTS:
        return PALETTE_LUTS[name]
    # Fallback: if default was deleted, use first available palette
    if DEFAULT_COLOR_PALETTE_NAME in PALETTE_LUTS:
        return PALETTE_LUTS[DEFAULT_COLOR_PALETTE_NAME]
    # If even default is gone, use any available palette
    if PALETTE_LUTS:
        return next(iter(PALETTE_LUTS.values()))
    # If no palettes at all, create a simple grayscale fallback
    grayscale = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    return _build_rgb_lut_from_stops(_rgb_colors_to_stops(grayscale))


def set_palette_spec(name: str, spec: dict, *, persist: bool = True) -> None:
    """Add or update a palette spec in the user library.

    When persist is False, updates runtime palettes without writing to disk.
    """
    global _RUNTIME_PALETTE_SPECS, _RUNTIME_PALETTES, PALETTE_LUTS

    if persist:
        user_palettes = _load_user_palette_specs()
        user_palettes[name] = spec
        _save_user_palette_specs(user_palettes)

    if _palette_spec_is_deleted(spec):
        _RUNTIME_PALETTE_SPECS.pop(name, None)
        _RUNTIME_PALETTES.pop(name, None)
        PALETTE_LUTS.pop(name, None)
        return

    _RUNTIME_PALETTE_SPECS[name] = spec
    colors = _palette_spec_to_colormap_colors(spec)
    if colors.size == 0:
        _RUNTIME_PALETTES.pop(name, None)
    else:
        _RUNTIME_PALETTES[name] = colors
    PALETTE_LUTS[name] = _palette_spec_to_lut(spec)


def delete_palette(name: str):
    """Permanently delete a palette from user library.

    Note: You can delete default palettes, but they'll reappear if you delete
    the palettes.json file. The app will gracefully handle missing defaults.
    """
    set_palette_spec(name, {"deleted": True})
