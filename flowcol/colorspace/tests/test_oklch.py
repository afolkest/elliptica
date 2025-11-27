"""Tests for OKLCH color space conversions."""

import numpy as np
import pytest

from flowcol.colorspace import (
    oklch_to_srgb,
    srgb_to_oklch,
    oklch_to_oklab,
    oklab_to_oklch,
    linear_to_srgb,
    srgb_to_linear,
    is_in_gamut,
    gamut_clip,
    gamut_map_to_srgb,
    max_chroma_for_lh,
    max_chroma_fast,
)


class TestBasicConversions:
    """Test basic color space conversions."""

    def test_black(self):
        """Black: L=0 should give RGB (0,0,0)."""
        L = np.array([0.0])
        C = np.array([0.0])
        H = np.array([0.0])
        rgb = oklch_to_srgb(L, C, H)
        np.testing.assert_allclose(rgb, [[0, 0, 0]], atol=1e-6)

    def test_white(self):
        """White: L=1, C=0 should give RGB (1,1,1)."""
        L = np.array([1.0])
        C = np.array([0.0])
        H = np.array([0.0])
        rgb = oklch_to_srgb(L, C, H)
        np.testing.assert_allclose(rgb, [[1, 1, 1]], atol=1e-4)

    def test_gray(self):
        """Mid gray: L=0.5, C=0 should give neutral gray."""
        L = np.array([0.5])
        C = np.array([0.0])
        H = np.array([0.0])
        rgb = oklch_to_srgb(L, C, H)
        # Should be roughly equal R=G=B
        assert np.allclose(rgb[0, 0], rgb[0, 1], atol=1e-4)
        assert np.allclose(rgb[0, 1], rgb[0, 2], atol=1e-4)

    def test_hue_varies_color(self):
        """Different hues at same L,C should give different colors."""
        L = np.array([0.7, 0.7, 0.7, 0.7])
        C = np.array([0.15, 0.15, 0.15, 0.15])
        H = np.array([0, 90, 180, 270])  # Red-ish, yellow-ish, cyan-ish, blue-ish

        rgb = oklch_to_srgb(L, C, H)

        # Each color should be different
        assert not np.allclose(rgb[0], rgb[1])
        assert not np.allclose(rgb[1], rgb[2])
        assert not np.allclose(rgb[2], rgb[3])


class TestRoundTrip:
    """Test sRGB -> OKLCH -> sRGB round trips."""

    def test_roundtrip_primaries(self):
        """Primary colors should round-trip accurately."""
        colors = np.array([
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
        ], dtype=np.float32)

        L, C, H = srgb_to_oklch(colors)
        rgb_back = oklch_to_srgb(L, C, H)
        rgb_back = np.clip(rgb_back, 0, 1)

        np.testing.assert_allclose(rgb_back, colors, atol=1e-4)

    def test_roundtrip_grays(self):
        """Grayscale values should round-trip accurately."""
        grays = np.array([
            [0, 0, 0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75],
            [1, 1, 1],
        ], dtype=np.float32)

        L, C, H = srgb_to_oklch(grays)
        rgb_back = oklch_to_srgb(L, C, H)
        rgb_back = np.clip(rgb_back, 0, 1)

        np.testing.assert_allclose(rgb_back, grays, atol=1e-4)

    def test_roundtrip_random(self):
        """Random in-gamut colors should round-trip."""
        rng = np.random.default_rng(42)
        colors = rng.random((100, 3)).astype(np.float32)

        L, C, H = srgb_to_oklch(colors)
        rgb_back = oklch_to_srgb(L, C, H)
        rgb_back = np.clip(rgb_back, 0, 1)

        np.testing.assert_allclose(rgb_back, colors, atol=1e-3)


class TestGammaEncoding:
    """Test sRGB gamma encoding/decoding."""

    def test_gamma_roundtrip(self):
        """Linear -> sRGB -> Linear should round-trip."""
        linear = np.linspace(0, 1, 100).astype(np.float32)
        srgb = linear_to_srgb(linear)
        linear_back = srgb_to_linear(srgb)
        np.testing.assert_allclose(linear_back, linear, atol=1e-6)

    def test_gamma_threshold(self):
        """Values near threshold should be handled correctly."""
        # Around the 0.0031308 threshold
        linear = np.array([0.001, 0.003, 0.0031308, 0.004, 0.01])
        srgb = linear_to_srgb(linear)
        linear_back = srgb_to_linear(srgb)
        np.testing.assert_allclose(linear_back, linear, atol=1e-6)


class TestOklabOklch:
    """Test OKLab <-> OKLCH conversions."""

    def test_oklch_to_oklab_zero_chroma(self):
        """Zero chroma should give a=b=0."""
        L = np.array([0.5])
        C = np.array([0.0])
        H = np.array([123.0])  # Hue shouldn't matter

        L_out, a, b = oklch_to_oklab(L, C, H)

        assert L_out[0] == 0.5
        np.testing.assert_allclose(a, [0], atol=1e-10)
        np.testing.assert_allclose(b, [0], atol=1e-10)

    def test_oklab_oklch_roundtrip(self):
        """OKLab -> OKLCH -> OKLab should round-trip."""
        L = np.array([0.7])
        a = np.array([0.1])
        b = np.array([-0.05])

        L2, C, H = oklab_to_oklch(L, a, b)
        L3, a2, b2 = oklch_to_oklab(L2, C, H)

        np.testing.assert_allclose(L3, L, atol=1e-10)
        np.testing.assert_allclose(a2, a, atol=1e-10)
        np.testing.assert_allclose(b2, b, atol=1e-10)


class TestGamut:
    """Test gamut checking and mapping."""

    def test_in_gamut_grays(self):
        """Neutral grays should always be in gamut."""
        L = np.linspace(0, 1, 10)
        C = np.zeros(10)
        H = np.zeros(10)

        assert is_in_gamut(L, C, H).all()

    def test_out_of_gamut_high_chroma(self):
        """Very high chroma should be out of gamut."""
        L = np.array([0.5])
        C = np.array([0.5])  # Way too high
        H = np.array([0.0])

        assert not is_in_gamut(L, C, H).all()

    def test_gamut_clip_stays_valid(self):
        """Clipped values should be in [0,1]."""
        L = np.array([0.5, 0.9, 0.1])
        C = np.array([0.4, 0.3, 0.3])  # Some out of gamut
        H = np.array([0, 120, 240])

        rgb = gamut_clip(L, C, H)

        assert (rgb >= 0).all()
        assert (rgb <= 1).all()

    def test_gamut_map_compress(self):
        """Compressed gamut mapping should preserve L and H."""
        L = np.array([0.7])
        C = np.array([0.4])  # Out of gamut
        H = np.array([30.0])

        rgb = gamut_map_to_srgb(L, C, H, method='compress')

        assert (rgb >= 0).all()
        assert (rgb <= 1).all()

        # Convert back and check L, H preserved
        L2, C2, H2 = srgb_to_oklch(rgb)
        np.testing.assert_allclose(L2, L, atol=0.01)
        np.testing.assert_allclose(H2, H, atol=1.0)  # Hue within 1 degree


class TestMaxChroma:
    """Test max chroma computation."""

    def test_max_chroma_black_white(self):
        """At L=0 and L=1, max chroma should be very small."""
        L = np.array([0.0, 1.0])
        H = np.array([0.0, 180.0])

        max_c = max_chroma_for_lh(L, H)
        # At extreme lightness, max chroma is very limited
        # L=0 (black) allows tiny chroma due to numerical precision
        # L=1 (white) allows essentially zero
        assert max_c[0] < 0.05  # Black: very limited
        assert max_c[1] < 0.01  # White: essentially zero

    def test_max_chroma_mid_lightness(self):
        """At L=0.5-0.7, there should be significant chroma headroom."""
        L = np.array([0.6])
        H = np.array([180.0])

        max_c = max_chroma_for_lh(L, H)
        assert max_c[0] > 0.1  # Should have decent chroma available

    def test_max_chroma_fast_matches_slow(self):
        """Fast LUT lookup should match binary search."""
        L = np.array([0.3, 0.5, 0.7, 0.9])
        H = np.array([0, 90, 180, 270])

        max_c_slow = max_chroma_for_lh(L, H, steps=20)
        max_c_fast = max_chroma_fast(L, H)

        np.testing.assert_allclose(max_c_fast, max_c_slow, atol=0.01)


class TestArrayShapes:
    """Test that various array shapes work correctly."""

    def test_2d_arrays(self):
        """2D arrays (images) should work."""
        shape = (64, 64)
        L = np.full(shape, 0.7)
        C = np.full(shape, 0.1)
        H = np.linspace(0, 360, shape[1])[None, :] * np.ones((shape[0], 1))

        rgb = gamut_map_to_srgb(L, C, H)

        assert rgb.shape == (64, 64, 3)
        assert (rgb >= 0).all()
        assert (rgb <= 1).all()

    def test_scalar_like(self):
        """Single-element arrays should work."""
        L = np.array(0.5)
        C = np.array(0.1)
        H = np.array(180.0)

        rgb = oklch_to_srgb(L, C, H)
        assert rgb.shape == (3,)


class TestTorchBackend:
    """Test torch tensor support (if torch available)."""

    @pytest.fixture
    def torch(self):
        pytest.importorskip('torch')
        import torch
        return torch

    def test_torch_basic(self, torch):
        """Basic torch tensor support."""
        L = torch.tensor([0.7])
        C = torch.tensor([0.1])
        H = torch.tensor([180.0])

        rgb = oklch_to_srgb(L, C, H)

        assert isinstance(rgb, torch.Tensor)
        assert rgb.shape == (1, 3)

    def test_torch_gpu_if_available(self, torch):
        """GPU tensors should stay on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        L = torch.tensor([0.7], device='cuda')
        C = torch.tensor([0.1], device='cuda')
        H = torch.tensor([180.0], device='cuda')

        rgb = oklch_to_srgb(L, C, H)

        assert rgb.device.type == 'cuda'

    def test_torch_numpy_parity(self, torch):
        """Torch and numpy should give same results."""
        L_np = np.array([0.3, 0.5, 0.7])
        C_np = np.array([0.1, 0.15, 0.1])
        H_np = np.array([0, 120, 240])

        L_t = torch.tensor(L_np)
        C_t = torch.tensor(C_np)
        H_t = torch.tensor(H_np)

        rgb_np = oklch_to_srgb(L_np, C_np, H_np)
        rgb_t = oklch_to_srgb(L_t, C_t, H_t)

        np.testing.assert_allclose(rgb_t.numpy(), rgb_np, atol=1e-5)
