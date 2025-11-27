"""Tests for project serialization (save/load)."""

import tempfile
from pathlib import Path
import numpy as np

from elliptica.app.core import AppState, RenderSettings, DisplaySettings, ConductorColorSettings, RegionStyle
from elliptica.types import Project, Conductor
from elliptica.serialization import save_project, load_project
from elliptica import defaults


def test_roundtrip_empty_project():
    """Test save/load with empty project."""
    state = AppState()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.elliptica"
        save_project(state, str(filepath))

        assert filepath.exists()

        loaded_state = load_project(str(filepath))

        # Verify project
        assert loaded_state.project.canvas_resolution == state.project.canvas_resolution
        assert loaded_state.project.streamlength_factor == state.project.streamlength_factor
        assert len(loaded_state.project.conductors) == 0

        # Verify settings
        assert loaded_state.render_settings.multiplier == state.render_settings.multiplier
        assert loaded_state.display_settings.gamma == state.display_settings.gamma


def test_roundtrip_single_conductor():
    """Test save/load with single conductor."""
    mask = np.random.rand(100, 100).astype(np.float32)
    conductor = Conductor(mask=mask, voltage=0.7, position=(50.0, 60.0))

    state = AppState()
    state.project.conductors.append(conductor)
    state.project.next_conductor_id = 1
    conductor.id = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.elliptica"
        save_project(state, str(filepath))

        loaded_state = load_project(str(filepath))

        assert len(loaded_state.project.conductors) == 1
        c = loaded_state.project.conductors[0]

        # Check scalar properties
        assert c.voltage == 0.7
        assert c.position == (50.0, 60.0)
        assert c.id == 0

        # Check mask roundtrip (uint16 PNG has precision loss)
        assert c.mask.shape == mask.shape
        assert np.allclose(c.mask, mask, atol=1.0/65535.0)


def test_roundtrip_multiple_conductors_with_interior():
    """Test save/load with multiple conductors including interior masks."""
    mask1 = np.random.rand(80, 80).astype(np.float32)
    interior1 = np.random.rand(80, 80).astype(np.float32)
    conductor1 = Conductor(
        mask=mask1,
        voltage=0.3,
        position=(10.0, 20.0),
        interior_mask=interior1,
        scale_factor=1.5,
        edge_smooth_sigma=2.0,
        id=0,
    )

    mask2 = np.random.rand(120, 120).astype(np.float32)
    conductor2 = Conductor(
        mask=mask2,
        voltage=0.9,
        position=(100.0, 150.0),
        edge_smooth_sigma=0.0,
        smear_enabled=True,
        smear_sigma=0.003,  # Fractional format (0.3% of canvas width)
        id=1,
    )

    state = AppState()
    state.project.conductors = [conductor1, conductor2]
    state.project.next_conductor_id = 2
    state.project.canvas_resolution = (1024, 768)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.elliptica"
        save_project(state, str(filepath))

        loaded_state = load_project(str(filepath))

        assert len(loaded_state.project.conductors) == 2
        assert loaded_state.project.canvas_resolution == (1024, 768)
        assert loaded_state.project.next_conductor_id == 2

        # Check conductor 1
        c1 = loaded_state.project.conductors[0]
        assert c1.voltage == 0.3
        assert c1.position == (10.0, 20.0)
        assert c1.scale_factor == 1.5
        assert c1.edge_smooth_sigma == 2.0
        assert c1.id == 0
        assert c1.mask.shape == mask1.shape
        assert c1.interior_mask is not None
        assert c1.interior_mask.shape == interior1.shape
        assert np.allclose(c1.interior_mask, interior1, atol=1.0/65535.0)

        # Check conductor 2
        c2 = loaded_state.project.conductors[1]
        assert c2.voltage == 0.9
        assert c2.position == (100.0, 150.0)
        assert c2.edge_smooth_sigma == 0.0
        assert c2.smear_enabled is True
        assert c2.smear_sigma == 0.003
        assert c2.id == 1
        assert c2.mask.shape == mask2.shape
        assert c2.interior_mask is None


def test_roundtrip_all_settings():
    """Test save/load with non-default settings."""
    state = AppState()

    # Custom render settings
    state.render_settings = RenderSettings(
        multiplier=2.0,
        supersample=2.0,
        num_passes=5,
        margin=0.15,
        noise_seed=42,
        noise_sigma=0.8,
    )

    # Custom display settings
    state.display_settings = DisplaySettings(
        downsample_sigma=0.7,
        clip_percent=0.03,
        contrast=1.5,
        gamma=1.2,
        color_enabled=True,
        palette="Viridis",
    )

    # Custom boundary conditions
    state.project.boundary_top = 1
    state.project.boundary_left = 1
    state.project.streamlength_factor = 3.5

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.elliptica"
        save_project(state, str(filepath))

        loaded_state = load_project(str(filepath))

        # Verify render settings
        rs = loaded_state.render_settings
        assert rs.multiplier == 2.0
        assert rs.supersample == 2.0
        assert rs.num_passes == 5
        assert rs.margin == 0.15
        assert rs.noise_seed == 42
        assert rs.noise_sigma == 0.8

        # Verify display settings
        ds = loaded_state.display_settings
        assert ds.downsample_sigma == 0.7
        assert ds.clip_percent == 0.03
        assert ds.contrast == 1.5
        assert ds.gamma == 1.2
        assert ds.color_enabled is True
        assert ds.palette == "Viridis"

        # Verify project settings
        assert loaded_state.project.boundary_top == 1
        assert loaded_state.project.boundary_left == 1
        assert loaded_state.project.streamlength_factor == 3.5


def test_roundtrip_conductor_color_settings():
    """Test save/load with conductor color settings."""
    mask = np.random.rand(50, 50).astype(np.float32)
    conductor = Conductor(mask=mask, voltage=0.5, id=0)

    state = AppState()
    state.project.conductors.append(conductor)
    state.project.next_conductor_id = 1

    # Add custom color settings
    color_settings = ConductorColorSettings(
        surface=RegionStyle(
            enabled=True,
            use_palette=False,
            palette="Plasma",
            solid_color=(0.8, 0.2, 0.3),
        ),
        interior=RegionStyle(
            enabled=True,
            use_palette=True,
            palette="Inferno",
            solid_color=(0.1, 0.5, 0.9),
        ),
    )
    state.conductor_color_settings[0] = color_settings

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.elliptica"
        save_project(state, str(filepath))

        loaded_state = load_project(str(filepath))

        assert 0 in loaded_state.conductor_color_settings
        cs = loaded_state.conductor_color_settings[0]

        # Check surface style
        assert cs.surface.enabled is True
        assert cs.surface.use_palette is False
        assert cs.surface.palette == "Plasma"
        assert cs.surface.solid_color == (0.8, 0.2, 0.3)

        # Check interior style
        assert cs.interior.enabled is True
        assert cs.interior.use_palette is True
        assert cs.interior.palette == "Inferno"
        assert cs.interior.solid_color == (0.1, 0.5, 0.9)


def test_backward_compatibility_missing_fields():
    """Test that missing fields in loaded data use defaults."""
    # This simulates loading an old file with new code that has added fields
    # We'll manually create a minimal metadata JSON and verify defaults are applied

    import json
    import zipfile

    mask = (np.random.rand(40, 40) > 0.5).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "old_format.flowcol"

        # Create minimal metadata (simulating old schema)
        metadata = {
            'schema_version': '1.0',
            'created_at': '2025-01-01T00:00:00',
            'project': {
                'canvas_resolution': [640, 480],
                'streamlength_factor': 2.0,
                'next_conductor_id': 1,
                # Missing boundary conditions (should default to 0)
            },
            'render_settings': {
                'multiplier': 1.0,
                # Missing other fields (should use defaults)
            },
            'display_settings': {
                'gamma': 1.0,
                # Missing other fields (should use defaults)
            },
            'conductors': [
                {
                    'voltage': 0.5,
                    'position': [0.0, 0.0],
                    'id': 0,
                    # Missing new fields like smear_enabled (should default to False)
                    'masks': {
                        'mask': {
                            'file': 'conductor_0_mask.png',
                            'shape': [40, 40],
                            'encoding': 'uint16_png',
                        },
                        'interior_mask': None,
                        'original_mask': None,
                        'original_interior_mask': None,
                    }
                }
            ],
            'conductor_color_settings': {},
        }

        # Create ZIP manually
        with zipfile.ZipFile(filepath, 'w') as zf:
            zf.writestr('metadata.json', json.dumps(metadata))

            # Save mask
            from PIL import Image
            from io import BytesIO
            mask_uint16 = (mask * 65535).astype(np.uint16)
            img = Image.fromarray(mask_uint16)
            buf = BytesIO()
            img.save(buf, format='PNG')
            zf.writestr('conductor_0_mask.png', buf.getvalue())

        # Load and verify defaults are applied
        loaded_state = load_project(str(filepath))

        # Project defaults
        assert loaded_state.project.boundary_top == 0
        assert loaded_state.project.boundary_bottom == 0
        assert loaded_state.project.boundary_left == 0
        assert loaded_state.project.boundary_right == 0

        # Render settings defaults
        assert loaded_state.render_settings.supersample == defaults.SUPERSAMPLE_CHOICES[0]
        assert loaded_state.render_settings.num_passes == defaults.DEFAULT_RENDER_PASSES

        # Display settings defaults
        assert loaded_state.display_settings.contrast == defaults.DEFAULT_CONTRAST

        # Conductor defaults
        c = loaded_state.project.conductors[0]
        assert c.smear_enabled is False
        assert c.edge_smooth_sigma == 1.5  # New default value
        assert c.scale_factor == 1.0


def test_file_extension_handling():
    """Test that .elliptica extension is added if missing."""
    from elliptica.serialization import save_project

    state = AppState()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save without extension
        filepath_no_ext = Path(tmpdir) / "myproject"
        save_project(state, str(filepath_no_ext))

        # Should create .elliptica file (new default extension)
        expected_path = Path(tmpdir) / "myproject.elliptica"
        assert expected_path.exists()

        # Should be loadable
        loaded_state = load_project(str(expected_path))
        assert loaded_state is not None


def test_render_cache_roundtrip():
    """Test saving and loading render cache."""
    from elliptica.serialization import save_render_cache, load_render_cache, compute_project_fingerprint
    from elliptica.app.core import RenderCache
    from elliptica.pipeline import RenderResult

    # Create a simple project
    mask = np.random.rand(100, 100).astype(np.float32)
    conductor = Conductor(mask=mask, voltage=0.5, position=(50.0, 60.0), id=0)

    project = Project()
    project.conductors.append(conductor)
    project.canvas_resolution = (200, 200)

    # Create fake render result
    lic_array = np.random.rand(200, 200).astype(np.float32)
    ex_array = np.random.randn(200, 200).astype(np.float32)  # Signed
    ey_array = np.random.randn(200, 200).astype(np.float32)  # Signed

    result = RenderResult(
        array=lic_array,
        compute_resolution=(200, 200),
        canvas_scaled_shape=(200, 200),
        margin=0.1,
        offset_x=0,
        offset_y=0,
        ex=ex_array,
        ey=ey_array,
    )

    cache = RenderCache(
        result=result,
        multiplier=1.0,
        supersample=1.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test.elliptica.cache"

        # Save cache
        save_render_cache(cache, project, str(cache_path))
        assert cache_path.exists()

        # Load cache
        loaded_cache = load_render_cache(str(cache_path), project)

        assert loaded_cache is not None
        assert loaded_cache.multiplier == 1.0
        assert loaded_cache.supersample == 1.0

        # Check arrays roundtrip correctly
        assert loaded_cache.result.array.shape == lic_array.shape
        assert np.allclose(loaded_cache.result.array, lic_array, atol=1.0/65535.0)

        assert loaded_cache.result.ex is not None
        assert np.allclose(loaded_cache.result.ex, ex_array, rtol=2e-3, atol=2e-3)  # uint16 precision

        assert loaded_cache.result.ey is not None
        assert np.allclose(loaded_cache.result.ey, ey_array, rtol=2e-3, atol=2e-3)  # uint16 precision

        # Check fingerprint
        expected_fp = compute_project_fingerprint(project)
        assert loaded_cache.project_fingerprint == expected_fp


def test_render_cache_fingerprint_detection():
    """Test that fingerprint detects project changes."""
    from elliptica.serialization import compute_project_fingerprint

    mask = np.ones((50, 50), dtype=np.float32)
    conductor = Conductor(mask=mask, voltage=0.5, position=(10.0, 20.0), id=0)

    project = Project()
    project.conductors.append(conductor)

    fp1 = compute_project_fingerprint(project)

    # Move conductor
    conductor.position = (15.0, 25.0)
    fp2 = compute_project_fingerprint(project)

    # Fingerprints should be different
    assert fp1 != fp2

    # Reset position
    conductor.position = (10.0, 20.0)
    fp3 = compute_project_fingerprint(project)

    # Should match original
    assert fp1 == fp3
