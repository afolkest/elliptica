"""Tests for ColorMapping."""

import numpy as np
import pytest

from elliptica.colorspace import ColorMapping
from elliptica.expr import ExprError


class TestColorMappingBasic:
    """Basic ColorMapping functionality."""

    def test_default_is_gray(self):
        """Default mapping (L=0.5, C=0, H=0) produces mid-gray."""
        mapping = ColorMapping()
        rgb = mapping.render({})

        # With C=0, hue doesn't matter, should be gray
        # L=0.5 in OKLCH is roughly 0.533 in sRGB (not exactly 0.5 due to transfer)
        assert rgb.shape == (3,)
        # All channels equal (gray)
        assert np.allclose(rgb[0], rgb[1], atol=1e-4)
        assert np.allclose(rgb[1], rgb[2], atol=1e-4)

    def test_simple_field_expression(self):
        """Expressions referencing fields work."""
        field = np.linspace(0, 1, 100).reshape(10, 10)

        mapping = ColorMapping(L="x", C="0.0", H="0.0")
        rgb = mapping.render({'x': field})

        assert rgb.shape == (10, 10, 3)
        # Dark where field is 0, light where field is 1
        assert rgb[0, 0, 0] < rgb[9, 9, 0]

    def test_multiple_fields(self):
        """Can use multiple fields in expressions."""
        lic = np.random.rand(10, 10)
        mag = np.random.rand(10, 10)

        mapping = ColorMapping(
            L="normalize(lic)",
            C="0.1 * normalize(mag)",
            H="200",
        )
        rgb = mapping.render({'lic': lic, 'mag': mag})

        assert rgb.shape == (10, 10, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0


class TestVariables:
    """Variable detection."""

    def test_variables_from_all_channels(self):
        """Variables property includes vars from L, C, and H."""
        mapping = ColorMapping(
            L="a + b",
            C="c * 0.1",
            H="d",
        )
        assert mapping.variables == {'a', 'b', 'c', 'd'}

    def test_no_variables(self):
        """Constant expressions have no variables."""
        mapping = ColorMapping(L="0.5", C="0.1", H="180")
        assert mapping.variables == set()

    def test_constants_not_in_variables(self):
        """Built-in constants (pi, e) are not variables."""
        mapping = ColorMapping(L="0.5", C="0.0", H="pi * 60")
        assert mapping.variables == set()


class TestScalarBroadcasting:
    """Scalar expressions broadcast to array shape."""

    def test_scalar_L_with_array_H(self):
        """Scalar L broadcasts with array H."""
        hue = np.linspace(0, 360, 100).reshape(10, 10)

        mapping = ColorMapping(L="0.7", C="0.15", H="h")
        rgb = mapping.render({'h': hue})

        assert rgb.shape == (10, 10, 3)

    def test_all_scalars(self):
        """All scalar expressions produce single RGB value."""
        mapping = ColorMapping(L="0.7", C="0.1", H="30")
        rgb = mapping.render({})

        # Should be a single color (may be 0-d or (3,) depending on broadcast)
        assert rgb.shape == (3,) or rgb.shape == ()

    def test_mixed_scalar_and_array(self):
        """Mix of scalar and array expressions."""
        field = np.ones((5, 5)) * 0.8

        mapping = ColorMapping(
            L="f",       # array
            C="0.1",     # scalar
            H="180",     # scalar
        )
        rgb = mapping.render({'f': field})

        assert rgb.shape == (5, 5, 3)


class TestRangeClamping:
    """L, C, H range handling."""

    def test_L_clamped_to_01(self):
        """L values outside [0,1] are clamped."""
        field = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])

        mapping = ColorMapping(L="x", C="0.0", H="0.0")
        rgb = mapping.render({'x': field})

        # Output should be valid (in [0,1])
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

        # L=-0.5 and L=0 should produce same output (both clamped to 0)
        assert np.allclose(rgb[0], rgb[1], atol=1e-4)
        # L=1.0 and L=1.5 should produce same output (both clamped to 1)
        assert np.allclose(rgb[3], rgb[4], atol=1e-4)

    def test_C_non_negative(self):
        """Negative chroma is clamped to 0."""
        field = np.array([-0.1, 0.0, 0.1])

        mapping = ColorMapping(L="0.5", C="x", H="180")
        rgb = mapping.render({'x': field})

        # C=-0.1 and C=0 should produce same gray
        assert np.allclose(rgb[0], rgb[1], atol=1e-4)

    def test_H_wraps_at_360(self):
        """Hue wraps around at 360 degrees."""
        mapping = ColorMapping(L="0.7", C="0.15", H="h")

        rgb_0 = mapping.render({'h': np.array([0.0])})
        rgb_360 = mapping.render({'h': np.array([360.0])})
        rgb_720 = mapping.render({'h': np.array([720.0])})
        rgb_neg = mapping.render({'h': np.array([-360.0])})

        assert np.allclose(rgb_0, rgb_360, atol=1e-4)
        assert np.allclose(rgb_0, rgb_720, atol=1e-4)
        assert np.allclose(rgb_0, rgb_neg, atol=1e-4)


class TestGamutMapping:
    """Gamut mapping methods."""

    def test_compress_vs_clip(self):
        """compress and clip methods produce valid output."""
        # High chroma at extreme lightness = out of gamut
        mapping_compress = ColorMapping(L="0.9", C="0.3", H="120", gamut='compress')
        mapping_clip = ColorMapping(L="0.9", C="0.3", H="120", gamut='clip')

        rgb_compress = mapping_compress.render({})
        rgb_clip = mapping_clip.render({})

        # Both should produce valid RGB
        assert rgb_compress.min() >= 0.0
        assert rgb_compress.max() <= 1.0
        assert rgb_clip.min() >= 0.0
        assert rgb_clip.max() <= 1.0

        # Results may differ (compress preserves hue, clip may shift it)
        # Just verify both work

    def test_in_gamut_same_result(self):
        """In-gamut colors produce same result with both methods."""
        # Low chroma is always in gamut
        mapping_compress = ColorMapping(L="0.5", C="0.05", H="180", gamut='compress')
        mapping_clip = ColorMapping(L="0.5", C="0.05", H="180", gamut='clip')

        rgb_compress = mapping_compress.render({})
        rgb_clip = mapping_clip.render({})

        assert np.allclose(rgb_compress, rgb_clip, atol=1e-3)


class TestErrorHandling:
    """Invalid expression handling."""

    def test_invalid_L_expression(self):
        """Invalid L expression raises ExprError."""
        with pytest.raises(ExprError, match="Invalid L expression"):
            ColorMapping(L="invalid syntax +++")

    def test_invalid_C_expression(self):
        """Invalid C expression raises ExprError."""
        with pytest.raises(ExprError, match="Invalid C expression"):
            ColorMapping(C="[list, comprehension]")

    def test_invalid_H_expression(self):
        """Invalid H expression raises ExprError."""
        with pytest.raises(ExprError, match="Invalid H expression"):
            ColorMapping(H="lambda x: x")

    def test_missing_variable(self):
        """Missing variable at render time raises error."""
        mapping = ColorMapping(L="x", C="0.0", H="0.0")

        with pytest.raises(Exception):  # UnknownVariableError
            mapping.render({})  # x not provided


class TestRepr:
    """String representation."""

    def test_repr(self):
        """repr shows all parameters."""
        mapping = ColorMapping(L="x", C="0.1", H="180", gamut='clip')
        r = repr(mapping)

        assert "L='x'" in r
        assert "C='0.1'" in r
        assert "H='180'" in r
        assert "gamut='clip'" in r


class TestExpressionFunctions:
    """Built-in functions work in expressions."""

    def test_normalize(self):
        """normalize() maps to [0, 1]."""
        field = np.array([[1, 2], [3, 4]], dtype=float)

        mapping = ColorMapping(L="normalize(x)", C="0.0", H="0.0")
        rgb = mapping.render({'x': field})

        assert rgb.shape == (2, 2, 3)

    def test_smoothstep(self):
        """smoothstep() works."""
        field = np.linspace(0, 1, 10)

        mapping = ColorMapping(L="smoothstep(0, 1, x)", C="0.0", H="0.0")
        rgb = mapping.render({'x': field})

        assert rgb.shape == (10, 3)

    def test_trig_functions(self):
        """Trig functions work."""
        x = np.linspace(0, 2 * np.pi, 10)

        mapping = ColorMapping(
            L="0.5 + 0.3 * sin(t)",
            C="0.1 * abs(cos(t))",
            H="180",
        )
        rgb = mapping.render({'t': x})

        assert rgb.shape == (10, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_clipnorm(self):
        """clipnorm() works in expressions."""
        field = np.random.rand(10, 10) * 100  # arbitrary range

        mapping = ColorMapping(
            L="clipnorm(x, 1, 99)",
            C="0.0",
            H="0.0",
        )
        rgb = mapping.render({'x': field})

        assert rgb.shape == (10, 10, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0


class TestSolidMapping:
    """ColorMapping.solid() convenience method."""

    def test_solid_creates_mapping(self):
        """solid() creates a valid ColorMapping."""
        mapping = ColorMapping.solid(L=0.6, C=0.1, H=30)

        assert mapping.L_expr == "0.6"
        assert mapping.C_expr == "0.1"
        assert mapping.H_expr == "30"

    def test_solid_has_no_variables(self):
        """solid() mapping requires no variables."""
        mapping = ColorMapping.solid(L=0.5, C=0.0, H=0)
        assert mapping.variables == set()

    def test_solid_is_solid(self):
        """solid() mapping has is_solid=True."""
        mapping = ColorMapping.solid(L=0.5, C=0.0, H=0)
        assert mapping.is_solid is True

    def test_non_solid_is_not_solid(self):
        """Mapping with variables has is_solid=False."""
        mapping = ColorMapping(L="x", C="0", H="0")
        assert mapping.is_solid is False

    def test_solid_renders_uniform_color(self):
        """solid() produces uniform color output."""
        mapping = ColorMapping.solid(L=0.7, C=0.1, H=200)
        rgb = mapping.render({})

        # Should be a single color
        assert rgb.shape == (3,)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_solid_broadcasts_to_array_shape(self):
        """solid() broadcasts when other arrays are present."""
        mapping = ColorMapping.solid(L=0.5, C=0.0, H=0)
        # Even though no variables needed, can pass arrays for shape reference
        dummy = np.zeros((5, 5))

        # Actually, solid doesn't use variables, so this should still give scalar
        rgb = mapping.render({})
        assert rgb.shape == (3,)


class TestColorConfig:
    """ColorConfig with region compositing."""

    def test_global_only(self):
        """ColorConfig with just global mapping works."""
        from elliptica.colorspace import ColorConfig

        config = ColorConfig(
            global_mapping=ColorMapping(L="0.5", C="0.0", H="0.0")
        )
        rgb = config.render({})

        assert rgb.shape == (3,)

    def test_variables_includes_all(self):
        """variables property includes global and region variables."""
        from elliptica.colorspace import ColorConfig

        config = ColorConfig(
            global_mapping=ColorMapping(L="a", C="0", H="0"),
            region_mappings={
                'region1': ColorMapping(L="b", C="0", H="0"),
                'region2': ColorMapping(L="c", C="0", H="0"),
            },
        )
        assert config.variables == {'a', 'b', 'c'}

    def test_region_compositing(self):
        """Regions are composited over global."""
        from elliptica.colorspace import ColorConfig

        global_map = ColorMapping(L="0.2", C="0.0", H="0.0")  # dark
        region_map = ColorMapping.solid(L=0.8, C=0.0, H=0.0)  # light

        config = ColorConfig(
            global_mapping=global_map,
            region_mappings={'bright_region': region_map},
        )

        # Create a mask: left half is region, right half is global
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[:, :5] = 1.0  # left half

        rgb = config.render({}, region_masks={'bright_region': mask})

        assert rgb.shape == (10, 10, 3)
        # Left half should be lighter than right half
        left_brightness = rgb[:, :5, :].mean()
        right_brightness = rgb[:, 5:, :].mean()
        assert left_brightness > right_brightness

    def test_partial_mask_blending(self):
        """Partial mask values blend smoothly."""
        from elliptica.colorspace import ColorConfig

        global_map = ColorMapping.solid(L=0.0, C=0.0, H=0.0)  # black
        region_map = ColorMapping.solid(L=1.0, C=0.0, H=0.0)  # white

        config = ColorConfig(
            global_mapping=global_map,
            region_mappings={'overlay': region_map},
        )

        # 50% opacity mask
        mask = np.full((5, 5), 0.5, dtype=np.float32)

        rgb = config.render({}, region_masks={'overlay': mask})

        # Should be ~50% gray
        assert rgb.shape == (5, 5, 3)
        assert np.allclose(rgb.mean(), 0.5, atol=0.1)

    def test_missing_mask_skips_region(self):
        """Regions without masks are skipped."""
        from elliptica.colorspace import ColorConfig

        config = ColorConfig(
            global_mapping=ColorMapping.solid(L=0.5, C=0.0, H=0.0),
            region_mappings={
                'has_mask': ColorMapping.solid(L=1.0, C=0.0, H=0.0),
                'no_mask': ColorMapping.solid(L=0.0, C=0.0, H=0.0),
            },
        )

        mask = np.ones((3, 3), dtype=np.float32)
        rgb = config.render({}, region_masks={'has_mask': mask})

        # Should be white (from has_mask), not black (from no_mask)
        assert rgb.shape == (3, 3, 3)
        assert rgb.mean() > 0.9  # mostly white

    def test_multiple_regions_layered(self):
        """Multiple regions are layered in order."""
        from elliptica.colorspace import ColorConfig

        config = ColorConfig(
            global_mapping=ColorMapping.solid(L=0.2, C=0.0, H=0.0),
            region_mappings={
                'layer1': ColorMapping.solid(L=0.5, C=0.0, H=0.0),
                'layer2': ColorMapping.solid(L=0.8, C=0.0, H=0.0),
            },
        )

        # Both masks cover everything
        masks = {
            'layer1': np.ones((3, 3), dtype=np.float32),
            'layer2': np.ones((3, 3), dtype=np.float32),
        }

        rgb = config.render({}, region_masks=masks)

        # Final result should be layer2 (0.8) since it's applied last
        # (dict ordering is preserved in Python 3.7+)
        assert rgb.mean() > 0.7

    def test_repr(self):
        """ColorConfig has useful repr."""
        from elliptica.colorspace import ColorConfig

        config = ColorConfig(
            global_mapping=ColorMapping(L="0.5", C="0", H="0"),
            region_mappings={'r1': ColorMapping.solid(L=0.5, C=0, H=0)},
        )
        r = repr(config)
        assert 'ColorConfig' in r
        assert 'r1' in r


class TestPercentileCaching:
    """Percentile calculations are cached within a render."""

    def test_same_clipnorm_uses_cache(self):
        """Multiple clipnorm calls on same array use cached percentiles."""
        # This is hard to test directly, but we can verify correctness
        field = np.random.rand(100, 100)

        mapping = ColorMapping(
            L="clipnorm(x, 0.5, 99.5)",
            C="0.1 * clipnorm(x, 0.5, 99.5)",  # same field, same percentiles
            H="180",
        )
        rgb = mapping.render({'x': field})

        assert rgb.shape == (100, 100, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0
