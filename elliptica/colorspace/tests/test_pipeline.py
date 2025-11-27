"""Tests for colorspace pipeline integration."""

import pytest
import numpy as np
import torch

from elliptica.colorspace import (
    ColorMapping,
    ColorConfig,
    build_field_bindings,
    build_region_masks,
    render_with_color_config,
    render_with_color_config_gpu,
)


class TestBuildFieldBindings:
    """Tests for build_field_bindings()."""

    def test_lic_only(self):
        """Test with only LIC field."""
        lic = np.random.rand(10, 10).astype(np.float32)
        bindings = build_field_bindings(lic)

        assert 'lic' in bindings
        assert np.array_equal(bindings['lic'], lic)
        assert 'mag' not in bindings
        assert 'ex' not in bindings
        assert 'ey' not in bindings

    def test_with_field_components(self):
        """Test with ex, ey field components."""
        lic = np.random.rand(10, 10).astype(np.float32)
        ex = np.random.rand(10, 10).astype(np.float32)
        ey = np.random.rand(10, 10).astype(np.float32)

        bindings = build_field_bindings(lic, ex, ey)

        assert 'lic' in bindings
        assert 'mag' in bindings
        assert 'ex' in bindings
        assert 'ey' in bindings

        # Verify magnitude calculation
        expected_mag = np.sqrt(ex**2 + ey**2)
        np.testing.assert_allclose(bindings['mag'], expected_mag)

    def test_with_solution_dict(self):
        """Test with additional solution fields."""
        lic = np.random.rand(10, 10).astype(np.float32)
        solution = {
            'phi': np.random.rand(10, 10).astype(np.float32),
            'E': np.random.rand(10, 10).astype(np.float32),
        }

        bindings = build_field_bindings(lic, solution=solution)

        assert 'lic' in bindings
        assert 'phi' in bindings
        assert 'E' in bindings
        assert np.array_equal(bindings['phi'], solution['phi'])

    def test_solution_does_not_override_standard(self):
        """Test that solution fields don't override standard names."""
        lic = np.random.rand(10, 10).astype(np.float32)
        ex = np.random.rand(10, 10).astype(np.float32)
        ey = np.random.rand(10, 10).astype(np.float32)
        solution = {
            'lic': np.zeros((10, 10)),  # Should NOT override
            'extra': np.ones((10, 10)),  # Should be added
        }

        bindings = build_field_bindings(lic, ex, ey, solution=solution)

        # lic should be the original, not from solution
        assert np.array_equal(bindings['lic'], lic)
        assert 'extra' in bindings

    def test_with_torch_tensors(self):
        """Test that torch tensors work correctly."""
        lic = torch.rand(10, 10)
        ex = torch.rand(10, 10)
        ey = torch.rand(10, 10)

        bindings = build_field_bindings(lic, ex, ey)

        assert torch.is_tensor(bindings['lic'])
        assert torch.is_tensor(bindings['mag'])
        expected_mag = torch.sqrt(ex**2 + ey**2)
        torch.testing.assert_close(bindings['mag'], expected_mag)


class MockConductor:
    """Mock conductor for testing."""
    def __init__(self, id: int):
        self.id = id


class TestBuildRegionMasks:
    """Tests for build_region_masks()."""

    def test_basic_masks(self):
        """Test basic mask construction."""
        conductors = [MockConductor(0), MockConductor(1)]
        surface_masks = [np.ones((10, 10)), np.zeros((10, 10))]
        interior_masks = [np.zeros((10, 10)), np.ones((10, 10))]

        masks = build_region_masks(surface_masks, interior_masks, conductors)

        assert 'conductor_0_surface' in masks
        assert 'conductor_0_interior' in masks
        assert 'conductor_1_surface' in masks
        assert 'conductor_1_interior' in masks

    def test_none_masks_skipped(self):
        """Test that None masks are skipped."""
        conductors = [MockConductor(0)]
        surface_masks = [None]
        interior_masks = [np.ones((10, 10))]

        masks = build_region_masks(surface_masks, interior_masks, conductors)

        assert 'conductor_0_surface' not in masks
        assert 'conductor_0_interior' in masks

    def test_empty_conductors(self):
        """Test with no conductors."""
        masks = build_region_masks([], [], [])
        assert masks == {}


class TestRenderWithColorConfig:
    """Tests for render_with_color_config()."""

    def test_simple_render(self):
        """Test simple render with constant expressions."""
        config = ColorConfig(
            global_mapping=ColorMapping(L="0.5", C="0.1", H="180"),
        )
        bindings = {'lic': np.random.rand(10, 10).astype(np.float32)}

        rgb = render_with_color_config(config, bindings)

        assert rgb.shape == (10, 10, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_expression_based_render(self):
        """Test render with expressions referencing fields."""
        config = ColorConfig(
            global_mapping=ColorMapping(
                L="0.3 + 0.4 * normalize(lic)",
                C="0.1",
                H="200",
            ),
        )
        lic = np.random.rand(10, 10).astype(np.float32)
        bindings = {'lic': lic}

        rgb = render_with_color_config(config, bindings)

        assert rgb.shape == (10, 10, 3)
        # Should have variation (not solid color)
        assert rgb.std() > 0

    def test_with_region_masks(self):
        """Test render with region overrides."""
        config = ColorConfig(
            global_mapping=ColorMapping(L="0.5", C="0.1", H="200"),
            region_mappings={
                'region_a': ColorMapping.solid(L=0.2, C=0.0, H=0),
            },
        )
        bindings = {'lic': np.ones((10, 10), dtype=np.float32)}
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[2:8, 2:8] = 1.0
        region_masks = {'region_a': mask}

        rgb = render_with_color_config(config, bindings, region_masks)

        assert rgb.shape == (10, 10, 3)
        # Region should have different color
        center = rgb[5, 5]
        edge = rgb[0, 0]
        assert not np.allclose(center, edge, atol=0.1)


class TestRenderWithColorConfigGPU:
    """Tests for render_with_color_config_gpu()."""

    def test_basic_gpu_render(self):
        """Test GPU render returns correct shape."""
        config = ColorConfig(
            global_mapping=ColorMapping(L="0.5", C="0.1", H="180"),
        )
        scalar_tensor = torch.rand(10, 10)

        rgb = render_with_color_config_gpu(config, scalar_tensor)

        assert rgb.shape == (10, 10, 3)
        assert torch.is_tensor(rgb)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_with_field_components(self):
        """Test with ex, ey field tensors."""
        config = ColorConfig(
            global_mapping=ColorMapping(
                L="0.3 + 0.4 * normalize(lic)",
                C="0.15 * normalize(mag)",
                H="180",
            ),
        )
        scalar_tensor = torch.rand(10, 10)
        ex_tensor = torch.rand(10, 10)
        ey_tensor = torch.rand(10, 10)

        rgb = render_with_color_config_gpu(
            config, scalar_tensor,
            ex_tensor=ex_tensor, ey_tensor=ey_tensor,
        )

        assert rgb.shape == (10, 10, 3)

    def test_with_conductor_masks(self):
        """Test with conductor masks."""
        config = ColorConfig(
            global_mapping=ColorMapping(L="0.5", C="0.1", H="180"),
            region_mappings={
                'conductor_0_interior': ColorMapping.solid(L=0.1, C=0, H=0),
            },
        )
        conductors = [MockConductor(0)]
        scalar_tensor = torch.rand(10, 10)
        interior_mask = torch.zeros(10, 10)
        interior_mask[3:7, 3:7] = 1.0

        rgb = render_with_color_config_gpu(
            config, scalar_tensor,
            interior_masks_gpu=[interior_mask],
            conductors=conductors,
        )

        assert rgb.shape == (10, 10, 3)
        # Center should be darker (interior region)
        center_lum = rgb[5, 5].mean()
        edge_lum = rgb[0, 0].mean()
        assert center_lum < edge_lum


class TestPostprocessIntegration:
    """Integration tests with apply_full_postprocess_gpu."""

    def test_color_config_path(self):
        """Test that color_config parameter works in postprocess pipeline."""
        from elliptica.gpu.postprocess import apply_full_postprocess_gpu

        config = ColorConfig(
            global_mapping=ColorMapping(
                L="0.3 + 0.4 * clipnorm(lic, 1, 99)",
                C="0.1",
                H="200",
            ),
        )

        scalar_tensor = torch.rand(20, 20)

        rgb, percentiles = apply_full_postprocess_gpu(
            scalar_tensor=scalar_tensor,
            conductor_masks_cpu=None,
            interior_masks_cpu=None,
            conductor_color_settings={},
            conductors=[],
            render_shape=(20, 20),
            canvas_resolution=(20, 20),
            clip_percent=1.0,
            brightness=0.0,
            contrast=1.0,
            gamma=1.0,
            color_enabled=True,
            palette="Ink Wash",
            color_config=config,
        )

        assert rgb.shape == (20, 20, 3)
        assert torch.is_tensor(rgb)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_legacy_path_still_works(self):
        """Test that legacy palette path still works when color_config is None."""
        from elliptica.gpu.postprocess import apply_full_postprocess_gpu

        scalar_tensor = torch.rand(20, 20)

        rgb, percentiles = apply_full_postprocess_gpu(
            scalar_tensor=scalar_tensor,
            conductor_masks_cpu=None,
            interior_masks_cpu=None,
            conductor_color_settings={},
            conductors=[],
            render_shape=(20, 20),
            canvas_resolution=(20, 20),
            clip_percent=1.0,
            brightness=0.0,
            contrast=1.0,
            gamma=1.0,
            color_enabled=True,
            palette="Ink Wash",
            color_config=None,  # Explicitly None
        )

        assert rgb.shape == (20, 20, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0
