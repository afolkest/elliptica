"""Built-in function registry for expressions."""

import numpy as np
from typing import Any

Array = Any

# Lazy torch reference
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# === NumPy implementations ===

def _np_lerp(a, b, t):
    return a + t * (b - a)


def _np_smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def _np_normalize(x):
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _np_pclip(x, lo_pct, hi_pct):
    lo = np.percentile(x, lo_pct)
    hi = np.percentile(x, hi_pct)
    return np.clip(x, lo, hi)


# === Torch implementations ===

def _torch_lerp(a, b, t):
    return a + t * (b - a)


def _torch_smoothstep(edge0, edge1, x):
    torch = _get_torch()
    t = torch.clamp((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def _torch_normalize(x):
    torch = _get_torch()
    lo, hi = x.min(), x.max()
    if hi == lo:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def _torch_pclip(x, lo_pct, hi_pct):
    torch = _get_torch()
    lo = torch.quantile(x.flatten().float(), lo_pct / 100)
    hi = torch.quantile(x.flatten().float(), hi_pct / 100)
    return torch.clamp(x, lo, hi)


# Function registry: name -> (numpy_func, torch_func, num_args)
FUNCTIONS: dict[str, tuple] = {
    # Pointwise math (1 arg)
    'sin': (np.sin, lambda x: _get_torch().sin(x), 1),
    'cos': (np.cos, lambda x: _get_torch().cos(x), 1),
    'tan': (np.tan, lambda x: _get_torch().tan(x), 1),
    'sqrt': (np.sqrt, lambda x: _get_torch().sqrt(x), 1),
    'abs': (np.abs, lambda x: _get_torch().abs(x), 1),
    'exp': (np.exp, lambda x: _get_torch().exp(x), 1),
    'log': (np.log, lambda x: _get_torch().log(x), 1),
    'log10': (np.log10, lambda x: _get_torch().log10(x), 1),

    # Multi-arg pointwise (3 args)
    'clamp': (
        lambda x, lo, hi: np.clip(x, lo, hi),
        lambda x, lo, hi: _get_torch().clamp(x, lo, hi),
        3
    ),
    'lerp': (_np_lerp, _torch_lerp, 3),
    'smoothstep': (_np_smoothstep, _torch_smoothstep, 3),

    # Reductions (1 arg -> scalar)
    'min': (lambda x: x.min(), lambda x: x.min(), 1),
    'max': (lambda x: x.max(), lambda x: x.max(), 1),
    'mean': (lambda x: x.mean(), lambda x: x.float().mean(), 1),
    'std': (lambda x: x.std(), lambda x: x.float().std(), 1),

    # Global transforms (array -> array)
    'normalize': (_np_normalize, _torch_normalize, 1),
    'pclip': (_np_pclip, _torch_pclip, 3),
}


# Built-in constants
CONSTANTS: dict[str, float] = {
    'pi': np.pi,
    'e': np.e,
    'tau': 2 * np.pi,
}


def get_function(name: str, use_torch: bool):
    """Get function implementation by name."""
    if name not in FUNCTIONS:
        return None
    np_fn, torch_fn, _ = FUNCTIONS[name]
    return torch_fn if use_torch else np_fn


def get_num_args(name: str) -> int | None:
    """Get expected number of arguments for function."""
    if name not in FUNCTIONS:
        return None
    return FUNCTIONS[name][2]
