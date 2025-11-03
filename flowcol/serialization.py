"""Project serialization - save/load flowcol projects as ZIP archives."""

from __future__ import annotations

import json
import zipfile
import hashlib
from pathlib import Path
from typing import Any
from dataclasses import asdict
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image

from flowcol.app.core import AppState, RenderSettings, DisplaySettings, ConductorColorSettings, RegionStyle, RenderCache
from flowcol.types import Project, Conductor
from flowcol.pipeline import RenderResult
from flowcol import defaults

SCHEMA_VERSION = "1.0"


def save_project(state: AppState, filepath: str) -> None:
    """Save project state to a .flowcol ZIP archive.

    Format:
        myproject.flowcol (ZIP containing:)
        ├── metadata.json          # All scalar/string data + schema version
        ├── conductor_0_mask.png
        ├── conductor_0_interior.png
        ├── conductor_0_original_mask.png
        └── ...

    Args:
        state: Application state to save
        filepath: Output path (should end with .flowcol)
    """
    filepath = Path(filepath)
    if filepath.suffix != '.flowcol':
        filepath = filepath.with_suffix('.flowcol')

    # Build metadata dictionary
    metadata = {
        'schema_version': SCHEMA_VERSION,
        'created_at': datetime.now().isoformat(),
        'project': _project_to_dict(state.project),
        'render_settings': _render_settings_to_dict(state.render_settings),
        'display_settings': _display_settings_to_dict(state.display_settings),
        'conductors': [],
        'conductor_color_settings': {},
    }

    # Create ZIP archive
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Save each conductor and its masks
        for i, conductor in enumerate(state.project.conductors):
            conductor_meta = _conductor_to_dict(conductor, i)
            metadata['conductors'].append(conductor_meta)

            # Save mask PNGs
            _save_mask_to_zip(zf, conductor.mask, conductor_meta['masks']['mask']['file'])
            if conductor.interior_mask is not None:
                _save_mask_to_zip(zf, conductor.interior_mask, conductor_meta['masks']['interior_mask']['file'])
            if conductor.original_mask is not None:
                _save_mask_to_zip(zf, conductor.original_mask, conductor_meta['masks']['original_mask']['file'])
            if conductor.original_interior_mask is not None:
                _save_mask_to_zip(zf, conductor.original_interior_mask, conductor_meta['masks']['original_interior_mask']['file'])

        # Save conductor color settings
        for conductor_id, color_settings in state.conductor_color_settings.items():
            metadata['conductor_color_settings'][str(conductor_id)] = _color_settings_to_dict(color_settings)

        # Write metadata JSON
        zf.writestr('metadata.json', json.dumps(metadata, indent=2))


def load_project(filepath: str) -> AppState:
    """Load project state from a .flowcol ZIP archive.

    Args:
        filepath: Path to .flowcol file

    Returns:
        Reconstructed AppState

    Raises:
        ValueError: If schema version is unsupported
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Project file not found: {filepath}")

    with zipfile.ZipFile(filepath, 'r') as zf:
        # Load metadata
        metadata = json.loads(zf.read('metadata.json'))

        # Check schema version
        schema_version = metadata.get('schema_version', '1.0')
        if schema_version != SCHEMA_VERSION:
            # Future: add migration logic here
            raise ValueError(f"Unsupported schema version: {schema_version}. Expected {SCHEMA_VERSION}")

        # Reconstruct project
        project = _dict_to_project(metadata['project'])

        # Load conductors
        for conductor_meta in metadata['conductors']:
            # Load masks from ZIP
            masks = {}
            for mask_name, mask_info in conductor_meta['masks'].items():
                if mask_info is not None:
                    masks[mask_name] = _load_mask_from_zip(zf, mask_info['file'])

            conductor = _dict_to_conductor(conductor_meta, masks)
            project.conductors.append(conductor)

        # Reconstruct state
        state = AppState(
            project=project,
            render_settings=_dict_to_render_settings(metadata['render_settings']),
            display_settings=_dict_to_display_settings(metadata['display_settings']),
        )

        # Load conductor color settings
        for conductor_id_str, color_settings_dict in metadata.get('conductor_color_settings', {}).items():
            conductor_id = int(conductor_id_str)
            state.conductor_color_settings[conductor_id] = _dict_to_color_settings(color_settings_dict)

        return state


# ============================================================================
# Conversion helpers
# ============================================================================

def _project_to_dict(project: Project) -> dict[str, Any]:
    """Convert Project to JSON-serializable dict."""
    return {
        'canvas_resolution': list(project.canvas_resolution),
        'streamlength_factor': project.streamlength_factor,
        'next_conductor_id': project.next_conductor_id,
        'boundary_top': project.boundary_top,
        'boundary_bottom': project.boundary_bottom,
        'boundary_left': project.boundary_left,
        'boundary_right': project.boundary_right,
    }


def _dict_to_project(data: dict[str, Any]) -> Project:
    """Reconstruct Project from dict."""
    return Project(
        conductors=[],  # Will be populated separately
        canvas_resolution=tuple(data['canvas_resolution']),
        streamlength_factor=data.get('streamlength_factor', defaults.DEFAULT_STREAMLENGTH_FACTOR),
        next_conductor_id=data.get('next_conductor_id', 0),
        boundary_top=data.get('boundary_top', 0),
        boundary_bottom=data.get('boundary_bottom', 0),
        boundary_left=data.get('boundary_left', 0),
        boundary_right=data.get('boundary_right', 0),
    )


def _conductor_to_dict(conductor: Conductor, index: int) -> dict[str, Any]:
    """Convert Conductor to JSON-serializable dict (without numpy arrays)."""
    mask_h, mask_w = conductor.mask.shape

    masks_meta = {
        'mask': {
            'file': f'conductor_{index}_mask.png',
            'shape': [mask_h, mask_w],
            'encoding': 'uint16_png',
        }
    }

    # Optional masks
    if conductor.interior_mask is not None:
        masks_meta['interior_mask'] = {
            'file': f'conductor_{index}_interior.png',
            'shape': list(conductor.interior_mask.shape),
            'encoding': 'uint16_png',
        }
    else:
        masks_meta['interior_mask'] = None

    if conductor.original_mask is not None:
        masks_meta['original_mask'] = {
            'file': f'conductor_{index}_original_mask.png',
            'shape': list(conductor.original_mask.shape),
            'encoding': 'uint16_png',
        }
    else:
        masks_meta['original_mask'] = None

    if conductor.original_interior_mask is not None:
        masks_meta['original_interior_mask'] = {
            'file': f'conductor_{index}_original_interior.png',
            'shape': list(conductor.original_interior_mask.shape),
            'encoding': 'uint16_png',
        }
    else:
        masks_meta['original_interior_mask'] = None

    return {
        'voltage': conductor.voltage,
        'position': list(conductor.position),
        'scale_factor': conductor.scale_factor,
        'edge_smooth_sigma': conductor.edge_smooth_sigma,
        'smear_enabled': conductor.smear_enabled,
        'smear_sigma': conductor.smear_sigma,
        'id': conductor.id,
        'masks': masks_meta,
    }


def _dict_to_conductor(data: dict[str, Any], masks: dict[str, np.ndarray]) -> Conductor:
    """Reconstruct Conductor from dict + loaded masks."""
    # Migrate old blur_sigma/blur_is_fractional to edge_smooth_sigma for backward compatibility
    if 'edge_smooth_sigma' in data:
        edge_smooth_sigma = data['edge_smooth_sigma']
    elif 'blur_sigma' in data:
        # Legacy migration
        blur_sigma = data.get('blur_sigma', 0.0)
        blur_is_fractional = data.get('blur_is_fractional', False)
        if blur_is_fractional:
            # Convert from fraction to pixels (assuming 1000px reference)
            edge_smooth_sigma = min(blur_sigma * 1000.0, 5.0)
        else:
            # Clamp pixel value to new 0-5 range
            edge_smooth_sigma = min(blur_sigma, 5.0)
    else:
        edge_smooth_sigma = 1.5

    # Migrate old absolute pixel smear_sigma to fractional (backward compatibility)
    smear_sigma_raw = data.get('smear_sigma', 0.002)
    if smear_sigma_raw > 0.1:
        # Old format: absolute pixels (range was 0.1-10.0)
        # Convert to fractional: assume 1024px canvas reference
        smear_sigma = smear_sigma_raw / 1024.0
    else:
        # New format: already fractional
        smear_sigma = smear_sigma_raw

    return Conductor(
        mask=masks['mask'],
        voltage=data.get('voltage', 0.5),
        position=tuple(data.get('position', [0.0, 0.0])),
        interior_mask=masks.get('interior_mask'),
        original_mask=masks.get('original_mask'),
        original_interior_mask=masks.get('original_interior_mask'),
        scale_factor=data.get('scale_factor', 1.0),
        edge_smooth_sigma=edge_smooth_sigma,
        smear_enabled=data.get('smear_enabled', False),
        smear_sigma=smear_sigma,
        id=data.get('id'),
    )


def _render_settings_to_dict(settings: RenderSettings) -> dict[str, Any]:
    """Convert RenderSettings to dict."""
    return {
        'multiplier': settings.multiplier,
        'supersample': settings.supersample,
        'num_passes': settings.num_passes,
        'margin': settings.margin,
        'noise_seed': settings.noise_seed,
        'noise_sigma': settings.noise_sigma,
        'use_mask': settings.use_mask,
        'edge_gain_strength': settings.edge_gain_strength,
        'edge_gain_power': settings.edge_gain_power,
        'poisson_scale': settings.poisson_scale,
    }


def _dict_to_render_settings(data: dict[str, Any]) -> RenderSettings:
    """Reconstruct RenderSettings from dict."""
    return RenderSettings(
        multiplier=data.get('multiplier', defaults.RENDER_RESOLUTION_CHOICES[0]),
        supersample=data.get('supersample', defaults.SUPERSAMPLE_CHOICES[0]),
        num_passes=data.get('num_passes', defaults.DEFAULT_RENDER_PASSES),
        margin=data.get('margin', defaults.DEFAULT_PADDING_MARGIN),
        noise_seed=data.get('noise_seed', defaults.DEFAULT_NOISE_SEED),
        noise_sigma=data.get('noise_sigma', defaults.DEFAULT_NOISE_SIGMA),
        use_mask=data.get('use_mask', defaults.DEFAULT_USE_MASK),
        edge_gain_strength=data.get('edge_gain_strength', defaults.DEFAULT_EDGE_GAIN_STRENGTH),
        edge_gain_power=data.get('edge_gain_power', defaults.DEFAULT_EDGE_GAIN_POWER),
        poisson_scale=max(
            defaults.MIN_POISSON_SCALE,
            min(defaults.MAX_POISSON_SCALE, data.get('poisson_scale', defaults.DEFAULT_POISSON_SCALE)),
        ),
    )


def _display_settings_to_dict(settings: DisplaySettings) -> dict[str, Any]:
    """Convert DisplaySettings to dict."""
    return {
        'downsample_sigma': settings.downsample_sigma,
        'clip_percent': settings.clip_percent,
        'brightness': settings.brightness,
        'contrast': settings.contrast,
        'gamma': settings.gamma,
        'color_enabled': settings.color_enabled,
        'palette': settings.palette,
    }


def _dict_to_display_settings(data: dict[str, Any]) -> DisplaySettings:
    """Reconstruct DisplaySettings from dict."""
    return DisplaySettings(
        downsample_sigma=data.get('downsample_sigma', defaults.DEFAULT_DOWNSAMPLE_SIGMA),
        clip_percent=data.get('clip_percent', defaults.DEFAULT_CLIP_PERCENT),
        brightness=data.get('brightness', defaults.DEFAULT_BRIGHTNESS),
        contrast=data.get('contrast', defaults.DEFAULT_CONTRAST),
        gamma=data.get('gamma', defaults.DEFAULT_GAMMA),
        color_enabled=data.get('color_enabled', defaults.DEFAULT_COLOR_ENABLED),
        palette=data.get('palette', defaults.DEFAULT_COLOR_PALETTE),
    )


def _color_settings_to_dict(settings: ConductorColorSettings) -> dict[str, Any]:
    """Convert ConductorColorSettings to dict."""
    return {
        'surface': _region_style_to_dict(settings.surface),
        'interior': _region_style_to_dict(settings.interior),
    }


def _dict_to_color_settings(data: dict[str, Any]) -> ConductorColorSettings:
    """Reconstruct ConductorColorSettings from dict."""
    return ConductorColorSettings(
        surface=_dict_to_region_style(data.get('surface', {})),
        interior=_dict_to_region_style(data.get('interior', {})),
    )


def _region_style_to_dict(style: RegionStyle) -> dict[str, Any]:
    """Convert RegionStyle to dict."""
    return {
        'enabled': style.enabled,
        'use_palette': style.use_palette,
        'palette': style.palette,
        'solid_color': list(style.solid_color),
    }


def _dict_to_region_style(data: dict[str, Any]) -> RegionStyle:
    """Reconstruct RegionStyle from dict."""
    return RegionStyle(
        enabled=data.get('enabled', False),
        use_palette=data.get('use_palette', True),
        palette=data.get('palette', defaults.DEFAULT_COLOR_PALETTE),
        solid_color=tuple(data.get('solid_color', [0.5, 0.5, 0.5])),
    )


# ============================================================================
# Mask I/O helpers
# ============================================================================

def _save_mask_to_zip(zf: zipfile.ZipFile, mask: np.ndarray, filename: str) -> None:
    """Save float32 mask as uint16 PNG into ZIP archive.

    Converts [0, 1] float32 → [0, 65535] uint16 for good precision.
    """
    # Clip to [0, 1] and convert to uint16
    mask_clipped = np.clip(mask, 0.0, 1.0)
    mask_uint16 = (mask_clipped * 65535).astype(np.uint16)

    # Save as PNG (grayscale 16-bit)
    img = Image.fromarray(mask_uint16)

    # Write to ZIP in-memory
    buf = BytesIO()
    img.save(buf, format='PNG')
    zf.writestr(filename, buf.getvalue())


def _load_mask_from_zip(zf: zipfile.ZipFile, filename: str) -> np.ndarray:
    """Load uint16 PNG from ZIP and convert to float32 mask.

    Converts [0, 65535] uint16 → [0, 1] float32.
    """
    # Read PNG from ZIP
    buf = BytesIO(zf.read(filename))
    img = Image.open(buf)

    # Convert to numpy and normalize to [0, 1]
    mask_uint16 = np.array(img)
    mask_float32 = (mask_uint16 / 65535.0).astype(np.float32)

    return mask_float32


# ============================================================================
# Project Fingerprinting
# ============================================================================

def compute_project_fingerprint(project: Project) -> str:
    """Compute hash of all properties that affect rendering.

    This fingerprint changes whenever conductors are moved, voltages change,
    canvas is resized, or any other render-affecting property changes.

    Returns:
        32-character MD5 hex string
    """
    parts = [
        # Canvas and global settings
        f"canvas:{project.canvas_resolution[0]}x{project.canvas_resolution[1]}",
        f"streamlen:{project.streamlength_factor}",
        f"bounds:{project.boundary_top},{project.boundary_bottom},{project.boundary_left},{project.boundary_right}",
    ]

    # Per-conductor state (order matters!)
    for i, c in enumerate(project.conductors):
        # Scalar properties
        parts.append(f"c{i}:v={c.voltage}")
        parts.append(f"c{i}:pos={c.position[0]:.2f},{c.position[1]:.2f}")
        parts.append(f"c{i}:scale={c.scale_factor}")
        parts.append(f"c{i}:edge_smooth={c.edge_smooth_sigma}")

        # Mask data hash (expensive but necessary)
        mask_hash = hashlib.md5(c.mask.tobytes()).hexdigest()[:8]
        parts.append(f"c{i}:mask={mask_hash}")

    # Combine and hash
    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()


# ============================================================================
# Render Cache Serialization
# ============================================================================

def save_render_cache(
    cache: 'RenderCache',
    project: Project,
    filepath: str,
) -> None:
    """Save render cache to .flowcol.cache file.

    Args:
        cache: RenderCache to save
        project: Current project (for fingerprinting)
        filepath: Output path (should end with .flowcol.cache)
    """

    filepath = Path(filepath)

    # Compute current fingerprint
    fingerprint = compute_project_fingerprint(project)

    # Build metadata
    metadata = {
        'project_fingerprint': fingerprint,
        'multiplier': cache.multiplier,
        'supersample': cache.supersample,
        'compute_resolution': list(cache.result.compute_resolution),
        'canvas_scaled_shape': list(cache.result.canvas_scaled_shape),
        'margin': cache.result.margin,
        'offset_x': cache.result.offset_x,
        'offset_y': cache.result.offset_y,
        'created_at': datetime.now().isoformat(),
    }

    # Create ZIP archive
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Save metadata
        zf.writestr('metadata.json', json.dumps(metadata, indent=2))

        # Save LIC result (unsigned, already in [0, 1])
        _save_mask_to_zip(zf, cache.result.array, 'lic_result.png')

        # Save E-fields (signed floats, need special handling)
        if cache.result.ex is not None:
            _save_signed_float_array(zf, cache.result.ex, 'field_ex.png')
        if cache.result.ey is not None:
            _save_signed_float_array(zf, cache.result.ey, 'field_ey.png')


def load_render_cache(
    filepath: str,
    project: Project,
) -> 'RenderCache | None':
    """Load render cache from .flowcol.cache file.

    Compares project fingerprint to detect staleness. If fingerprint doesn't match,
    still loads the cache but marks it as potentially stale.

    Args:
        filepath: Path to .flowcol.cache file
        project: Current project (for fingerprint comparison)

    Returns:
        RenderCache with project_fingerprint set, or None if load fails
    """

    filepath = Path(filepath)
    if not filepath.exists():
        return None

    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            # Load metadata
            metadata = json.loads(zf.read('metadata.json'))

            # Load LIC result
            lic = _load_mask_from_zip(zf, 'lic_result.png')

            # Load E-fields if present
            ex = None
            ey = None
            if 'field_ex.png' in zf.namelist():
                ex = _load_signed_float_array(zf, 'field_ex.png')
            if 'field_ey.png' in zf.namelist():
                ey = _load_signed_float_array(zf, 'field_ey.png')

            # Reconstruct RenderResult
            result = RenderResult(
                array=lic,
                compute_resolution=tuple(metadata['compute_resolution']),
                canvas_scaled_shape=tuple(metadata['canvas_scaled_shape']),
                margin=metadata['margin'],
                offset_x=metadata['offset_x'],
                offset_y=metadata['offset_y'],
                ex=ex,
                ey=ey,
            )

            # Create cache with fingerprint
            cache = RenderCache(
                result=result,
                multiplier=metadata['multiplier'],
                supersample=metadata['supersample'],
            )

            # Store fingerprint for staleness detection
            # (Will be added to RenderCache dataclass)
            cache.project_fingerprint = metadata['project_fingerprint']

            return cache

    except Exception as e:
        # Corrupt or incompatible cache - ignore silently
        print(f"Failed to load render cache: {e}")
        return None


def _save_signed_float_array(zf: zipfile.ZipFile, array: np.ndarray, filename: str) -> None:
    """Save signed float array by normalizing to [0, 1] range.

    Saves the value range as separate JSON so we can denormalize on load.
    """
    # Find value range
    vmin = float(array.min())
    vmax = float(array.max())

    # Normalize to [0, 1]
    if vmax - vmin > 1e-10:
        normalized = (array - vmin) / (vmax - vmin)
    else:
        # Constant array
        normalized = np.zeros_like(array)

    # Save as uint16 PNG
    _save_mask_to_zip(zf, normalized, filename)

    # Save range metadata
    range_filename = filename.replace('.png', '_range.json')
    range_data = {'vmin': vmin, 'vmax': vmax}
    zf.writestr(range_filename, json.dumps(range_data))


def _load_signed_float_array(zf: zipfile.ZipFile, filename: str) -> np.ndarray:
    """Load signed float array by denormalizing from [0, 1] range."""
    # Load normalized array
    normalized = _load_mask_from_zip(zf, filename)

    # Load range metadata
    range_filename = filename.replace('.png', '_range.json')
    range_data = json.loads(zf.read(range_filename))
    vmin = range_data['vmin']
    vmax = range_data['vmax']

    # Denormalize
    if vmax - vmin > 1e-10:
        array = normalized * (vmax - vmin) + vmin
    else:
        # Constant array
        array = np.full_like(normalized, vmin)

    return array.astype(np.float32)
