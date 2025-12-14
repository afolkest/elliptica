"""Project serialization - save/load Elliptica projects as ZIP archives."""

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

from elliptica.app.core import AppState, RenderSettings, DisplaySettings, BoundaryColorSettings, RegionStyle, RenderCache
from elliptica.types import Project, BoundaryObject
from elliptica.pipeline import RenderResult
from elliptica import defaults

SCHEMA_VERSION = "2.0"


def save_project(state: AppState, filepath: str) -> None:
    """Save project state to a .elliptica ZIP archive.

    Format:
        myproject.elliptica (ZIP containing:)
        ├── metadata.json          # All scalar/string data + schema version
        ├── boundary_0_mask.png
        ├── boundary_0_interior.png
        ├── boundary_0_original_mask.png
        └── ...

    Args:
        state: Application state to save
        filepath: Output path (should end with .elliptica)
    """
    filepath = Path(filepath)
    # Accept both .elliptica and legacy .flowcol extensions
    if filepath.suffix not in ('.elliptica', '.flowcol'):
        filepath = filepath.with_suffix('.elliptica')

    # Build metadata dictionary
    metadata = {
        'schema_version': SCHEMA_VERSION,
        'created_at': datetime.now().isoformat(),
        'project': _project_to_dict(state.project),
        'render_settings': _render_settings_to_dict(state.render_settings),
        'display_settings': _display_settings_to_dict(state.display_settings),
        'boundary_objects': [],
        'boundary_color_settings': {},
    }

    # Create ZIP archive
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Save each boundary object and its masks
        for i, boundary in enumerate(state.project.boundary_objects):
            boundary_meta = _boundary_object_to_dict(boundary, i)
            metadata['boundary_objects'].append(boundary_meta)

            # Save mask PNGs
            _save_mask_to_zip(zf, boundary.mask, boundary_meta['masks']['mask']['file'])
            if boundary.interior_mask is not None:
                _save_mask_to_zip(zf, boundary.interior_mask, boundary_meta['masks']['interior_mask']['file'])
            if boundary.original_mask is not None:
                _save_mask_to_zip(zf, boundary.original_mask, boundary_meta['masks']['original_mask']['file'])
            if boundary.original_interior_mask is not None:
                _save_mask_to_zip(zf, boundary.original_interior_mask, boundary_meta['masks']['original_interior_mask']['file'])

        # Save boundary color settings
        for boundary_id, color_settings in state.boundary_color_settings.items():
            metadata['boundary_color_settings'][str(boundary_id)] = _color_settings_to_dict(color_settings)

        # Write metadata JSON
        zf.writestr('metadata.json', json.dumps(metadata, indent=2))


class ProjectLoadError(Exception):
    """Error loading project file."""
    pass


def load_project(filepath: str) -> AppState:
    """Load project state from a .elliptica ZIP archive.

    Args:
        filepath: Path to .elliptica file

    Returns:
        Reconstructed AppState

    Raises:
        ProjectLoadError: If file is corrupt, wrong version, or missing data
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Project file not found: {filepath}")

    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            # Load metadata
            try:
                metadata = json.loads(zf.read('metadata.json'))
            except (KeyError, json.JSONDecodeError) as e:
                raise ProjectLoadError(f"Corrupt or missing metadata.json: {e}")

            # Check schema version
            schema_version = metadata.get('schema_version', '1.0')
            if schema_version != SCHEMA_VERSION:
                raise ProjectLoadError(
                    f"Schema version {schema_version} not supported. "
                    f"Expected {SCHEMA_VERSION}. Run migration script: python -m elliptica.migrate {filepath}"
                )

            # Reconstruct project
            project = _dict_to_project(metadata['project'])

            # Load boundary objects
            for boundary_meta in metadata['boundary_objects']:
                # Load masks from ZIP
                masks = {}
                for mask_name, mask_info in boundary_meta['masks'].items():
                    if mask_info is not None:
                        try:
                            masks[mask_name] = _load_mask_from_zip(zf, mask_info['file'])
                        except KeyError:
                            raise ProjectLoadError(f"Missing mask file: {mask_info['file']}")
                        except Exception as e:
                            raise ProjectLoadError(f"Corrupt mask file {mask_info['file']}: {e}")

                boundary = _dict_to_boundary_object(boundary_meta, masks)
                project.boundary_objects.append(boundary)

            # Reconstruct state
            state = AppState(
                project=project,
                render_settings=_dict_to_render_settings(metadata['render_settings']),
                display_settings=_dict_to_display_settings(metadata['display_settings']),
            )

            # Load boundary color settings
            for boundary_id_str, color_settings_dict in metadata.get('boundary_color_settings', {}).items():
                boundary_id = int(boundary_id_str)
                state.boundary_color_settings[boundary_id] = _dict_to_color_settings(color_settings_dict)

            return state

    except zipfile.BadZipFile:
        raise ProjectLoadError(f"Not a valid ZIP file: {filepath}")


# ============================================================================
# Conversion helpers
# ============================================================================

def _project_to_dict(project: Project) -> dict[str, Any]:
    """Convert Project to JSON-serializable dict."""
    return {
        'canvas_resolution': list(project.canvas_resolution),
        'streamlength_factor': project.streamlength_factor,
        'next_boundary_id': project.next_boundary_id,
        'boundary_top': project.boundary_top,
        'boundary_bottom': project.boundary_bottom,
        'boundary_left': project.boundary_left,
        'boundary_right': project.boundary_right,
        'pde_type': project.pde_type,
        'pde_params': project.pde_params,
        'pde_bc': project.pde_bc,
    }


def _dict_to_project(data: dict[str, Any]) -> Project:
    """Reconstruct Project from dict."""
    return Project(
        boundary_objects=[],  # Will be populated separately
        canvas_resolution=tuple(data['canvas_resolution']),
        streamlength_factor=data['streamlength_factor'],
        next_boundary_id=data['next_boundary_id'],
        boundary_top=data['boundary_top'],
        boundary_bottom=data['boundary_bottom'],
        boundary_left=data['boundary_left'],
        boundary_right=data['boundary_right'],
        pde_type=data['pde_type'],
        pde_params=data['pde_params'],
        pde_bc=data['pde_bc'],
    )


def _boundary_object_to_dict(boundary: BoundaryObject, index: int) -> dict[str, Any]:
    """Convert BoundaryObject to JSON-serializable dict (without numpy arrays)."""
    mask_h, mask_w = boundary.mask.shape

    masks_meta = {
        'mask': {
            'file': f'boundary_{index}_mask.png',
            'shape': [mask_h, mask_w],
            'encoding': 'uint16_png',
        }
    }

    # Optional masks
    if boundary.interior_mask is not None:
        masks_meta['interior_mask'] = {
            'file': f'boundary_{index}_interior.png',
            'shape': list(boundary.interior_mask.shape),
            'encoding': 'uint16_png',
        }
    else:
        masks_meta['interior_mask'] = None

    if boundary.original_mask is not None:
        masks_meta['original_mask'] = {
            'file': f'boundary_{index}_original_mask.png',
            'shape': list(boundary.original_mask.shape),
            'encoding': 'uint16_png',
        }
    else:
        masks_meta['original_mask'] = None

    if boundary.original_interior_mask is not None:
        masks_meta['original_interior_mask'] = {
            'file': f'boundary_{index}_original_interior.png',
            'shape': list(boundary.original_interior_mask.shape),
            'encoding': 'uint16_png',
        }
    else:
        masks_meta['original_interior_mask'] = None

    return {
        'params': boundary.params,
        'position': list(boundary.position),
        'scale_factor': boundary.scale_factor,
        'edge_smooth_sigma': boundary.edge_smooth_sigma,
        'smear_enabled': boundary.smear_enabled,
        'smear_sigma': boundary.smear_sigma,
        'id': boundary.id,
        'masks': masks_meta,
    }


def _dict_to_boundary_object(data: dict[str, Any], masks: dict[str, np.ndarray]) -> BoundaryObject:
    """Reconstruct BoundaryObject from dict + loaded masks."""
    return BoundaryObject(
        mask=masks['mask'],
        params=data['params'],
        position=tuple(data['position']),
        interior_mask=masks.get('interior_mask'),
        original_mask=masks.get('original_mask'),
        original_interior_mask=masks.get('original_interior_mask'),
        scale_factor=data['scale_factor'],
        edge_smooth_sigma=data['edge_smooth_sigma'],
        smear_enabled=data['smear_enabled'],
        smear_sigma=data['smear_sigma'],
        id=data['id'],
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
        'solve_scale': settings.solve_scale,
    }


def _dict_to_render_settings(data: dict[str, Any]) -> RenderSettings:
    """Reconstruct RenderSettings from dict."""
    return RenderSettings(
        multiplier=data['multiplier'],
        supersample=data['supersample'],
        num_passes=data['num_passes'],
        margin=data['margin'],
        noise_seed=data['noise_seed'],
        noise_sigma=data['noise_sigma'],
        use_mask=data['use_mask'],
        edge_gain_strength=data['edge_gain_strength'],
        edge_gain_power=data['edge_gain_power'],
        solve_scale=data['solve_scale'],
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
        'lightness_expr': settings.lightness_expr,
        'saturation': settings.saturation,
    }


def _dict_to_display_settings(data: dict[str, Any]) -> DisplaySettings:
    """Reconstruct DisplaySettings from dict."""
    return DisplaySettings(
        downsample_sigma=data['downsample_sigma'],
        clip_percent=data['clip_percent'],
        brightness=data['brightness'],
        contrast=data['contrast'],
        gamma=data['gamma'],
        color_enabled=data['color_enabled'],
        palette=data['palette'],
        lightness_expr=data['lightness_expr'],
        saturation=data['saturation'],
    )


def _color_settings_to_dict(settings: BoundaryColorSettings) -> dict[str, Any]:
    """Convert BoundaryColorSettings to dict."""
    return {
        'surface': _region_style_to_dict(settings.surface),
        'interior': _region_style_to_dict(settings.interior),
    }


def _dict_to_color_settings(data: dict[str, Any]) -> BoundaryColorSettings:
    """Reconstruct BoundaryColorSettings from dict."""
    return BoundaryColorSettings(
        surface=_dict_to_region_style(data['surface']),
        interior=_dict_to_region_style(data['interior']),
    )


def _region_style_to_dict(style: RegionStyle) -> dict[str, Any]:
    """Convert RegionStyle to dict."""
    return {
        'enabled': style.enabled,
        'use_palette': style.use_palette,
        'palette': style.palette,
        'solid_color': list(style.solid_color),
        'brightness': style.brightness,  # None or float
        'contrast': style.contrast,      # None or float
        'gamma': style.gamma,            # None or float
        'lightness_expr': style.lightness_expr,  # None or str
        'smear_enabled': style.smear_enabled,
        'smear_sigma': style.smear_sigma,
    }


def _dict_to_region_style(data: dict[str, Any]) -> RegionStyle:
    """Reconstruct RegionStyle from dict."""
    return RegionStyle(
        enabled=data['enabled'],
        use_palette=data['use_palette'],
        palette=data['palette'],
        solid_color=tuple(data['solid_color']),
        brightness=data['brightness'],  # May be None
        contrast=data['contrast'],      # May be None
        gamma=data['gamma'],            # May be None
        lightness_expr=data['lightness_expr'],  # May be None
        smear_enabled=data['smear_enabled'],
        smear_sigma=data['smear_sigma'],
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

    This fingerprint changes whenever boundary objects are moved, voltages change,
    canvas is resized, or any other render-affecting property changes.

    Returns:
        32-character MD5 hex string
    """
    parts = [
        # Canvas and global settings
        f"canvas:{project.canvas_resolution[0]}x{project.canvas_resolution[1]}",
        f"streamlen:{project.streamlength_factor}",
        f"bounds:{project.boundary_top},{project.boundary_bottom},{project.boundary_left},{project.boundary_right}",
        f"pde_type:{getattr(project, 'pde_type', 'poisson')}",
        f"pde_bc:{getattr(project, 'pde_bc', {})}",
    ]

    # Per-boundary state (order matters!)
    for i, b in enumerate(project.boundary_objects):
        # Scalar properties
        parts.append(f"b{i}:v={b.params.get('voltage', 0.0)}")
        parts.append(f"b{i}:pos={b.position[0]:.2f},{b.position[1]:.2f}")
        parts.append(f"b{i}:scale={b.scale_factor}")
        parts.append(f"b{i}:edge_smooth={b.edge_smooth_sigma}")

        # Mask data hash (expensive but necessary)
        mask_hash = hashlib.md5(b.mask.tobytes()).hexdigest()[:8]
        parts.append(f"b{i}:mask={mask_hash}")

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
    """Save render cache to .elliptica.cache file.

    Args:
        cache: RenderCache to save
        project: Current project (for fingerprinting)
        filepath: Output path (should end with .elliptica.cache or .flowcol.cache)
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

        # Save solution fields (phi, etc.) - signed floats
        if cache.result.solution:
            solution_keys = list(cache.result.solution.keys())
            metadata['solution_fields'] = solution_keys
            # Re-save metadata with solution_fields list
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
            for name, array in cache.result.solution.items():
                if isinstance(array, np.ndarray) and array.ndim == 2:
                    _save_signed_float_array(zf, array, f'solution_{name}.png')


def load_render_cache(
    filepath: str,
    project: Project,
) -> 'RenderCache | None':
    """Load render cache from .elliptica.cache or .flowcol.cache file.

    Compares project fingerprint to detect staleness. If fingerprint doesn't match,
    still loads the cache but marks it as potentially stale.

    Args:
        filepath: Path to .elliptica.cache or .flowcol.cache file
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

            # Load solution fields (phi, etc.) if present
            solution = None
            solution_fields = metadata.get('solution_fields', [])
            if solution_fields:
                solution = {}
                for name in solution_fields:
                    filename = f'solution_{name}.png'
                    if filename in zf.namelist():
                        solution[name] = _load_signed_float_array(zf, filename)

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
                solution=solution,
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
