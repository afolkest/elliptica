"""Project serialization - save/load flowcol projects as ZIP archives."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from dataclasses import asdict
from datetime import datetime

import numpy as np
from PIL import Image

from flowcol.app.core import AppState, RenderSettings, DisplaySettings, ConductorColorSettings, RegionStyle
from flowcol.types import Project, Conductor
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
        'blur_sigma': conductor.blur_sigma,
        'blur_is_fractional': conductor.blur_is_fractional,
        'smear_enabled': conductor.smear_enabled,
        'smear_sigma': conductor.smear_sigma,
        'smear_feather': conductor.smear_feather,
        'id': conductor.id,
        'masks': masks_meta,
    }


def _dict_to_conductor(data: dict[str, Any], masks: dict[str, np.ndarray]) -> Conductor:
    """Reconstruct Conductor from dict + loaded masks."""
    return Conductor(
        mask=masks['mask'],
        voltage=data.get('voltage', 0.5),
        position=tuple(data.get('position', [0.0, 0.0])),
        interior_mask=masks.get('interior_mask'),
        original_mask=masks.get('original_mask'),
        original_interior_mask=masks.get('original_interior_mask'),
        scale_factor=data.get('scale_factor', 1.0),
        blur_sigma=data.get('blur_sigma', 0.0),
        blur_is_fractional=data.get('blur_is_fractional', False),
        smear_enabled=data.get('smear_enabled', False),
        smear_sigma=data.get('smear_sigma', 2.0),
        smear_feather=data.get('smear_feather', 3.0),
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
    )


def _display_settings_to_dict(settings: DisplaySettings) -> dict[str, Any]:
    """Convert DisplaySettings to dict."""
    return {
        'downsample_sigma': settings.downsample_sigma,
        'clip_percent': settings.clip_percent,
        'contrast': settings.contrast,
        'gamma': settings.gamma,
        'color_enabled': settings.color_enabled,
        'palette': settings.palette,
        'edge_blur_sigma': settings.edge_blur_sigma,
        'edge_blur_falloff': settings.edge_blur_falloff,
        'edge_blur_strength': settings.edge_blur_strength,
    }


def _dict_to_display_settings(data: dict[str, Any]) -> DisplaySettings:
    """Reconstruct DisplaySettings from dict."""
    return DisplaySettings(
        downsample_sigma=data.get('downsample_sigma', defaults.DEFAULT_DOWNSAMPLE_SIGMA),
        clip_percent=data.get('clip_percent', defaults.DEFAULT_CLIP_PERCENT),
        contrast=data.get('contrast', defaults.DEFAULT_CONTRAST),
        gamma=data.get('gamma', defaults.DEFAULT_GAMMA),
        color_enabled=data.get('color_enabled', defaults.DEFAULT_COLOR_ENABLED),
        palette=data.get('palette', defaults.DEFAULT_COLOR_PALETTE),
        edge_blur_sigma=data.get('edge_blur_sigma', defaults.DEFAULT_EDGE_BLUR_SIGMA),
        edge_blur_falloff=data.get('edge_blur_falloff', defaults.DEFAULT_EDGE_BLUR_FALLOFF),
        edge_blur_strength=data.get('edge_blur_strength', defaults.DEFAULT_EDGE_BLUR_STRENGTH),
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
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format='PNG')
    zf.writestr(filename, buf.getvalue())


def _load_mask_from_zip(zf: zipfile.ZipFile, filename: str) -> np.ndarray:
    """Load uint16 PNG from ZIP and convert to float32 mask.

    Converts [0, 65535] uint16 → [0, 1] float32.
    """
    from io import BytesIO

    # Read PNG from ZIP
    buf = BytesIO(zf.read(filename))
    img = Image.open(buf)

    # Convert to numpy and normalize to [0, 1]
    mask_uint16 = np.array(img)
    mask_float32 = (mask_uint16 / 65535.0).astype(np.float32)

    return mask_float32
