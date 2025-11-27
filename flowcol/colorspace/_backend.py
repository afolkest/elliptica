"""Backend dispatch for numpy/torch compatibility.

Provides unified math operations that work with both numpy arrays and torch tensors.
Torch is imported lazily on first use to avoid loading it when not needed.
"""

import numpy as np
from typing import Any

Array = Any  # numpy.ndarray or torch.Tensor

# Lazy torch reference - only imported when needed
_torch = None


def _get_torch():
    """Get torch module, importing it on first use."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def is_torch(x: Array) -> bool:
    """Check if x is a torch tensor."""
    return type(x).__module__.startswith('torch')


def get_backend(x: Array):
    """Get the appropriate backend module for x."""
    if is_torch(x):
        return _get_torch()
    return np


# === Dispatched operations ===

def sin(x: Array) -> Array:
    if is_torch(x):
        return _get_torch().sin(x)
    return np.sin(x)


def cos(x: Array) -> Array:
    if is_torch(x):
        return _get_torch().cos(x)
    return np.cos(x)


def sqrt(x: Array) -> Array:
    if is_torch(x):
        return _get_torch().sqrt(x)
    return np.sqrt(x)


def cbrt(x: Array) -> Array:
    """Cube root (sign-preserving)."""
    if is_torch(x):
        torch = _get_torch()
        return torch.sign(x) * torch.abs(x).pow(1/3)
    return np.cbrt(x)


def pow(x: Array, exp: float) -> Array:
    if is_torch(x):
        return _get_torch().pow(x, exp)
    return np.power(x, exp)


def clip(x: Array, lo: float, hi: float) -> Array:
    if is_torch(x):
        return _get_torch().clamp(x, lo, hi)
    return np.clip(x, lo, hi)


def where(cond: Array, true_val: Array, false_val: Array) -> Array:
    if is_torch(cond):
        return _get_torch().where(cond, true_val, false_val)
    return np.where(cond, true_val, false_val)


def stack(arrays: list[Array], axis: int = -1) -> Array:
    """Stack arrays along a new axis."""
    if is_torch(arrays[0]):
        return _get_torch().stack(arrays, dim=axis)
    return np.stack(arrays, axis=axis)


def atan2(y: Array, x: Array) -> Array:
    if is_torch(y):
        return _get_torch().atan2(y, x)
    return np.arctan2(y, x)


def minimum(x: Array, y: Array) -> Array:
    if is_torch(x):
        return _get_torch().minimum(x, y)
    return np.minimum(x, y)


def maximum(x: Array, y: Array) -> Array:
    if is_torch(x):
        return _get_torch().maximum(x, y)
    return np.maximum(x, y)


def zeros_like(x: Array) -> Array:
    if is_torch(x):
        return _get_torch().zeros_like(x)
    return np.zeros_like(x)


def full_like(x: Array, value: float) -> Array:
    if is_torch(x):
        return _get_torch().full_like(x, value)
    return np.full_like(x, value)


def all_along_axis(x: Array, axis: int) -> Array:
    """Check if all values are True along axis."""
    if is_torch(x):
        return _get_torch().all(x, dim=axis)
    return np.all(x, axis=axis)


def to_numpy(x: Array) -> np.ndarray:
    """Convert to numpy array (moves from GPU if needed)."""
    if is_torch(x):
        return x.detach().cpu().numpy()
    return x


def from_numpy(arr: np.ndarray, reference: Array) -> Array:
    """Convert numpy array to same type/device as reference."""
    if is_torch(reference):
        return _get_torch().from_numpy(arr).to(device=reference.device, dtype=reference.dtype)
    return arr
