"""Map field arrays to RGB via OKLCH expressions.

This module ties together the expression system and colorspace conversion,
allowing users to define how physical fields map to perceptually uniform color.

Example:
    from elliptica.colorspace import ColorMapping

    # Define color mapping with expressions
    mapping = ColorMapping(
        L="0.3 + 0.5 * clipnorm(lic, 0.5, 99.5)",  # Lightness from texture
        C="0.12 * clipnorm(mag, 1, 99)",            # Chroma from field magnitude
        H="200",                                    # Fixed hue
    )

    # Render to RGB
    rgb = mapping.render({'lic': lic_array, 'mag': mag_array})

    # Convenience for solid colors
    copper = ColorMapping.solid(L=0.6, C=0.1, H=50)
"""

from typing import Literal, Any
import numpy as np

from . import _backend as B
from ._backend import Array
from .gamut import gamut_map_to_srgb
from elliptica.expr import compile_expression, get_variables, ExprError, clear_percentile_cache


class ColorMapping:
    """Maps field arrays to RGB via OKLCH expressions.

    Each OKLCH channel (Lightness, Chroma, Hue) is defined by an expression
    that references field variables. Expressions can use built-in functions
    like normalize(), smoothstep(), sin(), etc.

    Attributes:
        L_expr: Expression for Lightness (should produce values in [0, 1])
        C_expr: Expression for Chroma (should produce values >= 0, typically < 0.4)
        H_expr: Expression for Hue (degrees, will be wrapped to [0, 360))
        gamut: Gamut mapping method ('compress' or 'clip')
        variables: Set of all variable names required by the expressions
    """

    def __init__(
        self,
        L: str = "0.5",
        C: str = "0.0",
        H: str = "0.0",
        gamut: Literal['compress', 'clip'] = 'compress',
    ):
        """Create a color mapping from OKLCH expressions.

        Args:
            L: Lightness expression (0-1, will be clamped)
            C: Chroma expression (>=0, excess is gamut-mapped)
            H: Hue expression (degrees, will wrap at 360)
            gamut: How to handle out-of-gamut colors:
                   'compress' - reduce chroma to fit (preserves L and H intent)
                   'clip' - hard-clip RGB (fast but may distort colors)

        Raises:
            ExprError: If any expression is invalid
        """
        self.L_expr = L
        self.C_expr = C
        self.H_expr = H
        self.gamut = gamut

        # Compile expressions (fail fast on invalid syntax)
        try:
            self._L_fn = compile_expression(L)
        except ExprError as e:
            raise ExprError(f"Invalid L expression: {e}") from e

        try:
            self._C_fn = compile_expression(C)
        except ExprError as e:
            raise ExprError(f"Invalid C expression: {e}") from e

        try:
            self._H_fn = compile_expression(H)
        except ExprError as e:
            raise ExprError(f"Invalid H expression: {e}") from e

        # Collect all required variables
        self._variables = (
            get_variables(L) |
            get_variables(C) |
            get_variables(H)
        )

        # Track if this is a solid color (no expressions, just constants)
        self._is_solid = len(self._variables) == 0

    @classmethod
    def solid(
        cls,
        L: float = 0.5,
        C: float = 0.0,
        H: float = 0.0,
        gamut: Literal['compress', 'clip'] = 'compress',
    ) -> 'ColorMapping':
        """Create a solid color mapping (no field dependencies).

        This is a convenience method for creating mappings that produce
        a single uniform color, useful for boundary interiors, etc.

        Args:
            L: Lightness value (0-1)
            C: Chroma value (0-~0.4)
            H: Hue in degrees (0-360)
            gamut: Gamut mapping method

        Returns:
            ColorMapping that produces a solid color
        """
        return cls(L=str(L), C=str(C), H=str(H), gamut=gamut)

    @property
    def is_solid(self) -> bool:
        """True if this mapping produces a solid color (no field dependencies)."""
        return self._is_solid

    @property
    def variables(self) -> set[str]:
        """Set of variable names required by all expressions."""
        return self._variables

    def render(self, bindings: dict[str, Array], _clear_cache: bool = True) -> Array:
        """Render field arrays to RGB image.

        Args:
            bindings: Dict mapping variable names to arrays.
                      All arrays must be broadcastable to a common shape.
            _clear_cache: Internal flag to control percentile cache clearing.
                          Set to False when called from ColorConfig to avoid
                          clearing cache between region renders.

        Returns:
            RGB array with shape (..., 3) and values in [0, 1].

        Raises:
            UnknownVariableError: If a required variable is missing from bindings.
        """
        if _clear_cache:
            clear_percentile_cache()

        # Evaluate expressions
        L = self._L_fn(bindings)
        C = self._C_fn(bindings)
        H = self._H_fn(bindings)

        # Broadcast to common shape (handles scalar expressions)
        L, C, H = B.broadcast_arrays(L, C, H)

        # Clamp/wrap to valid OKLCH ranges
        L = B.clip(L, 0.0, 1.0)
        C = B.maximum(C, B.zeros_like(C))  # C >= 0
        H = H % 360  # wrap hue

        # Convert to sRGB with gamut handling
        return gamut_map_to_srgb(L, C, H, method=self.gamut)

    def __repr__(self) -> str:
        return (
            f"ColorMapping(\n"
            f"    L={self.L_expr!r},\n"
            f"    C={self.C_expr!r},\n"
            f"    H={self.H_expr!r},\n"
            f"    gamut={self.gamut!r},\n"
            f")"
        )


class ColorConfig:
    """Complete color configuration with global mapping and region overrides.

    Manages a global ColorMapping plus per-region overrides that are composited
    using masks. Regions are rendered in order and alpha-blended over the base.

    Example:
        config = ColorConfig(
            global_mapping=ColorMapping(
                L="clipnorm(lic, 0.5, 99.5)",
                C="0.1 * clipnorm(mag, 1, 99)",
                H="200",
            ),
            region_mappings={
                'boundary_0_surface': ColorMapping(
                    L="0.4 + 0.3 * clipnorm(lic, 0.5, 99.5)",
                    C="0.08",
                    H="30",  # copper
                ),
                'boundary_0_interior': ColorMapping.solid(L=0.12, C=0, H=0),
            },
        )

        rgb = config.render(
            bindings={'lic': lic, 'mag': mag},
            region_masks={
                'boundary_0_surface': surface_mask,
                'boundary_0_interior': interior_mask,
            },
        )
    """

    def __init__(
        self,
        global_mapping: ColorMapping,
        region_mappings: dict[str, ColorMapping] | None = None,
    ):
        """Create a color configuration.

        Args:
            global_mapping: Base ColorMapping applied to all pixels
            region_mappings: Optional dict of region_name -> ColorMapping
                            for per-region color overrides
        """
        self.global_mapping = global_mapping
        self.region_mappings = region_mappings or {}

    @property
    def variables(self) -> set[str]:
        """All variable names required by global and region mappings."""
        result = self.global_mapping.variables.copy()
        for mapping in self.region_mappings.values():
            result |= mapping.variables
        return result

    def render(
        self,
        bindings: dict[str, Array],
        region_masks: dict[str, Array] | None = None,
    ) -> Array:
        """Render to RGB with region compositing.

        Args:
            bindings: Dict mapping variable names to arrays
            region_masks: Dict mapping region names to mask arrays (0-1 alpha).
                         Only regions present in both region_mappings and
                         region_masks will be rendered.

        Returns:
            RGB array with shape (..., 3) and values in [0, 1]
        """
        region_masks = region_masks or {}

        # Clear cache once at the start, not for each mapping
        clear_percentile_cache()

        # Determine target shape and backend from bindings or masks
        target_shape = None
        use_torch = False
        reference_tensor = None

        # First try bindings (most reliable source of shape and backend)
        for arr in bindings.values():
            if hasattr(arr, 'shape') and len(arr.shape) >= 2:
                target_shape = arr.shape[:2]
                use_torch = B.is_torch(arr)
                reference_tensor = arr
                break
        # Fall back to masks if bindings are all scalar
        if target_shape is None:
            for mask in region_masks.values():
                if hasattr(mask, 'shape') and len(mask.shape) >= 2:
                    target_shape = mask.shape
                    use_torch = B.is_torch(mask)
                    reference_tensor = mask
                    break

        # Render global mapping
        rgb = self.global_mapping.render(bindings, _clear_cache=False)

        # Broadcast rgb to target shape if needed (handles solid colors)
        if target_shape is not None and rgb.ndim == 1:
            if use_torch:
                import torch
                # Convert numpy to torch if needed
                if not B.is_torch(rgb):
                    rgb = torch.from_numpy(np.asarray(rgb)).to(
                        device=reference_tensor.device, dtype=reference_tensor.dtype
                    )
                rgb = rgb.expand(*target_shape, -1)
            else:
                rgb = np.broadcast_to(rgb, (*target_shape, 3))

        # Composite region overrides
        for region_name, mapping in self.region_mappings.items():
            if region_name not in region_masks:
                continue

            mask = region_masks[region_name]
            region_rgb = mapping.render(bindings, _clear_cache=False)

            # Broadcast region_rgb to match mask shape if needed
            if region_rgb.ndim == 1 and mask.ndim >= 2:
                if use_torch:
                    import torch
                    # Convert numpy to torch if needed
                    if not B.is_torch(region_rgb):
                        region_rgb = torch.from_numpy(np.asarray(region_rgb)).to(
                            device=reference_tensor.device, dtype=reference_tensor.dtype
                        )
                    region_rgb = region_rgb.expand(*mask.shape, -1)
                else:
                    region_rgb = np.broadcast_to(region_rgb, (*mask.shape, 3))

            # Alpha blend: rgb = rgb * (1 - mask) + region_rgb * mask
            # Expand mask to broadcast with RGB
            mask_expanded = mask[..., None] if mask.ndim == rgb.ndim - 1 else mask
            rgb = rgb * (1 - mask_expanded) + region_rgb * mask_expanded

        return rgb

    def __repr__(self) -> str:
        regions = ', '.join(self.region_mappings.keys())
        return f"ColorConfig(global=..., regions=[{regions}])"
