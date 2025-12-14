"""Migration script for Elliptica project files.

Converts v1 project files to v2 format.

Usage:
    python -m elliptica.migrate project.elliptica
    python -m elliptica.migrate ~/projects/*.elliptica
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from io import BytesIO
from pathlib import Path

from elliptica import defaults


def migrate_v1_to_v2(filepath: Path) -> bool:
    """Migrate a v1 project file to v2 format.

    Args:
        filepath: Path to .elliptica or .flowcol file

    Returns:
        True if migration was performed, False if skipped (already v2 or error)
    """
    if not filepath.exists():
        print(f"  SKIP: File not found: {filepath}")
        return False

    # Read the ZIP
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            metadata = json.loads(zf.read('metadata.json'))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
        print(f"  SKIP: Cannot read {filepath}: {e}")
        return False

    # Check version
    schema_version = metadata.get('schema_version', '1.0')
    if schema_version == '2.0':
        print(f"  SKIP: Already v2: {filepath}")
        return False
    if schema_version != '1.0':
        print(f"  SKIP: Unknown schema version {schema_version}: {filepath}")
        return False

    print(f"  Migrating {filepath}...")

    # Transform metadata
    new_metadata = _transform_metadata(metadata)

    # Build file rename map
    rename_map = _build_rename_map(metadata)

    # Create backup
    backup_path = filepath.with_suffix(filepath.suffix + '.v1.bak')
    shutil.copy2(filepath, backup_path)
    print(f"    Backup: {backup_path}")

    # Determine output path (normalize .flowcol → .elliptica)
    if filepath.suffix == '.flowcol':
        output_path = filepath.with_suffix('.elliptica')
    else:
        output_path = filepath

    # Create new ZIP with transformed content
    with zipfile.ZipFile(filepath, 'r') as old_zf:
        # Read all files into memory
        files = {}
        for name in old_zf.namelist():
            if name == 'metadata.json':
                continue
            new_name = rename_map.get(name, name)
            files[new_name] = old_zf.read(name)

    # Write new ZIP
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as new_zf:
        new_zf.writestr('metadata.json', json.dumps(new_metadata, indent=2))
        for name, data in files.items():
            new_zf.writestr(name, data)

    # Remove old .flowcol if we renamed to .elliptica
    if output_path != filepath:
        filepath.unlink()
        print(f"    Renamed: {filepath.name} → {output_path.name}")

    print(f"    Done: {output_path}")
    return True


def _transform_metadata(metadata: dict) -> dict:
    """Transform v1 metadata to v2 format."""
    # Start with a copy
    new_meta = {
        'schema_version': '2.0',
        'created_at': metadata.get('created_at', ''),
    }

    # Transform project
    old_project = metadata.get('project', {})
    new_meta['project'] = {
        'canvas_resolution': old_project.get('canvas_resolution', [800, 600]),
        'streamlength_factor': old_project.get('streamlength_factor', defaults.DEFAULT_STREAMLENGTH_FACTOR),
        'next_boundary_id': old_project.get('next_conductor_id', 0),
        'boundary_top': old_project.get('boundary_top', 0),
        'boundary_bottom': old_project.get('boundary_bottom', 0),
        'boundary_left': old_project.get('boundary_left', 0),
        'boundary_right': old_project.get('boundary_right', 0),
        'pde_type': old_project.get('pde_type', 'poisson'),
        'pde_params': old_project.get('pde_params', {}),
        'pde_bc': old_project.get('pde_bc', {}),
    }

    # Transform render settings
    old_render = metadata.get('render_settings', {})
    # Support legacy poisson_scale → solve_scale
    solve_scale = old_render.get('solve_scale', old_render.get('poisson_scale', defaults.DEFAULT_SOLVE_SCALE))
    new_meta['render_settings'] = {
        'multiplier': old_render.get('multiplier', defaults.RENDER_RESOLUTION_CHOICES[0]),
        'supersample': old_render.get('supersample', defaults.SUPERSAMPLE_CHOICES[0]),
        'num_passes': old_render.get('num_passes', defaults.DEFAULT_RENDER_PASSES),
        'margin': old_render.get('margin', defaults.DEFAULT_PADDING_MARGIN),
        'noise_seed': old_render.get('noise_seed', defaults.DEFAULT_NOISE_SEED),
        'noise_sigma': old_render.get('noise_sigma', defaults.DEFAULT_NOISE_SIGMA),
        'use_mask': old_render.get('use_mask', defaults.DEFAULT_USE_MASK),
        'edge_gain_strength': old_render.get('edge_gain_strength', defaults.DEFAULT_EDGE_GAIN_STRENGTH),
        'edge_gain_power': old_render.get('edge_gain_power', defaults.DEFAULT_EDGE_GAIN_POWER),
        'solve_scale': max(defaults.MIN_SOLVE_SCALE, min(defaults.MAX_SOLVE_SCALE, solve_scale)),
    }

    # Transform display settings
    old_display = metadata.get('display_settings', {})
    new_meta['display_settings'] = {
        'downsample_sigma': old_display.get('downsample_sigma', defaults.DEFAULT_DOWNSAMPLE_SIGMA),
        'clip_percent': old_display.get('clip_percent', defaults.DEFAULT_CLIP_PERCENT),
        'brightness': old_display.get('brightness', defaults.DEFAULT_BRIGHTNESS),
        'contrast': old_display.get('contrast', defaults.DEFAULT_CONTRAST),
        'gamma': old_display.get('gamma', defaults.DEFAULT_GAMMA),
        'color_enabled': old_display.get('color_enabled', defaults.DEFAULT_COLOR_ENABLED),
        'palette': old_display.get('palette', defaults.DEFAULT_COLOR_PALETTE),
        'lightness_expr': old_display.get('lightness_expr'),
        'saturation': old_display.get('saturation', 1.0),
    }

    # Transform conductors → boundary_objects
    new_meta['boundary_objects'] = []
    for i, old_conductor in enumerate(metadata.get('conductors', [])):
        new_boundary = _transform_conductor(old_conductor, i)
        new_meta['boundary_objects'].append(new_boundary)

    # Transform conductor_color_settings → boundary_color_settings
    new_meta['boundary_color_settings'] = {}
    for conductor_id, old_settings in metadata.get('conductor_color_settings', {}).items():
        new_meta['boundary_color_settings'][conductor_id] = _transform_color_settings(old_settings)

    return new_meta


def _transform_conductor(old: dict, index: int) -> dict:
    """Transform v1 conductor to v2 boundary_object."""
    # Migrate edge_smooth_sigma from legacy blur_sigma
    if 'edge_smooth_sigma' in old:
        edge_smooth_sigma = old['edge_smooth_sigma']
    elif 'blur_sigma' in old:
        blur_sigma = old.get('blur_sigma', 0.0)
        blur_is_fractional = old.get('blur_is_fractional', False)
        if blur_is_fractional:
            edge_smooth_sigma = min(blur_sigma * 1000.0, 5.0)
        else:
            edge_smooth_sigma = min(blur_sigma, 5.0)
    else:
        edge_smooth_sigma = 1.5

    # Migrate smear_sigma from absolute pixels to fractional
    smear_sigma_raw = old.get('smear_sigma', 0.002)
    if smear_sigma_raw > 0.1:
        # Old format: absolute pixels
        smear_sigma = smear_sigma_raw / 1024.0
    else:
        smear_sigma = smear_sigma_raw

    # Build params dict from voltage
    params = {'voltage': old.get('voltage', 0.5)}

    # Transform mask file references
    old_masks = old.get('masks', {})
    new_masks = {}

    for mask_name in ['mask', 'interior_mask', 'original_mask', 'original_interior_mask']:
        old_info = old_masks.get(mask_name)
        if old_info is not None:
            new_file = old_info['file'].replace(f'conductor_{index}_', f'boundary_{index}_')
            new_masks[mask_name] = {
                'file': new_file,
                'shape': old_info['shape'],
                'encoding': old_info.get('encoding', 'uint16_png'),
            }
        else:
            new_masks[mask_name] = None

    return {
        'params': params,
        'position': old.get('position', [0.0, 0.0]),
        'scale_factor': old.get('scale_factor', 1.0),
        'edge_smooth_sigma': edge_smooth_sigma,
        'smear_enabled': old.get('smear_enabled', False),
        'smear_sigma': smear_sigma,
        'id': old.get('id'),
        'masks': new_masks,
    }


def _transform_color_settings(old: dict) -> dict:
    """Transform v1 color settings to v2 format."""
    return {
        'surface': _transform_region_style(old.get('surface', {})),
        'interior': _transform_region_style(old.get('interior', {})),
    }


def _transform_region_style(old: dict) -> dict:
    """Transform v1 region style to v2 format."""
    return {
        'enabled': old.get('enabled', False),
        'use_palette': old.get('use_palette', True),
        'palette': old.get('palette', defaults.DEFAULT_COLOR_PALETTE),
        'solid_color': old.get('solid_color', [0.5, 0.5, 0.5]),
        'brightness': old.get('brightness'),
        'contrast': old.get('contrast'),
        'gamma': old.get('gamma'),
        'lightness_expr': old.get('lightness_expr'),
        'smear_enabled': old.get('smear_enabled', False),
        'smear_sigma': old.get('smear_sigma', defaults.DEFAULT_SMEAR_SIGMA),
    }


def _build_rename_map(metadata: dict) -> dict[str, str]:
    """Build a map of old filenames to new filenames."""
    rename_map = {}

    for i, conductor in enumerate(metadata.get('conductors', [])):
        masks = conductor.get('masks', {})
        for mask_name, mask_info in masks.items():
            if mask_info is not None:
                old_file = mask_info['file']
                new_file = old_file.replace(f'conductor_{i}_', f'boundary_{i}_')
                if old_file != new_file:
                    rename_map[old_file] = new_file

    return rename_map


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m elliptica.migrate <file.elliptica> [file2.elliptica ...]")
        print("\nMigrates v1 project files to v2 format.")
        print("Original files are backed up to *.v1.bak")
        sys.exit(1)

    files = [Path(arg) for arg in sys.argv[1:]]
    migrated = 0
    skipped = 0

    print(f"Processing {len(files)} file(s)...")

    for filepath in files:
        if migrate_v1_to_v2(filepath):
            migrated += 1
        else:
            skipped += 1

    print(f"\nComplete: {migrated} migrated, {skipped} skipped")


if __name__ == '__main__':
    main()
