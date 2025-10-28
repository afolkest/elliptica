"""GPU acceleration using PyTorch MPS backend."""

import numpy as np
import torch
from typing import Optional


class GPUContext:
    """Global GPU context for MPS-accelerated operations."""

    _device: Optional[torch.device] = None
    _available: Optional[bool] = None

    @classmethod
    def device(cls) -> torch.device:
        """Get GPU device (lazy initialization)."""
        if cls._device is None:
            if cls.is_available():
                cls._device = torch.device('mps')
            else:
                cls._device = torch.device('cpu')
        return cls._device

    @classmethod
    def is_available(cls) -> bool:
        """Check if MPS GPU acceleration is available."""
        if cls._available is None:
            cls._available = torch.backends.mps.is_available()
        return cls._available

    @classmethod
    def warmup(cls) -> None:
        """Pre-compile Metal shaders with dummy operations (~500ms)."""
        if not cls.is_available():
            return

        device = cls.device()
        # Warmup with realistic operation sizes
        dummy = torch.randn(1024, 1024, device=device, dtype=torch.float32)

        # Gaussian blur warmup (torchvision will compile shaders)
        from torchvision.transforms.functional import gaussian_blur
        _ = gaussian_blur(dummy.unsqueeze(0).unsqueeze(0), kernel_size=5, sigma=2.0)

        # Percentile/quantile warmup
        _ = torch.quantile(dummy, torch.tensor([0.01, 0.99], device=device))

        # Arithmetic operations warmup
        _ = dummy * 1.5 + 0.5
        _ = torch.pow(dummy.clamp(0, 1), 1.2)

        # Synchronize to ensure all operations complete
        torch.mps.synchronize()

    @classmethod
    def to_gpu(cls, arr: np.ndarray) -> torch.Tensor:
        """Upload NumPy array to GPU as float32 tensor."""
        tensor = torch.from_numpy(arr).to(dtype=torch.float32)
        return tensor.to(cls.device())

    @classmethod
    def to_cpu(cls, tensor: torch.Tensor) -> np.ndarray:
        """Download GPU tensor to NumPy array."""
        return tensor.cpu().numpy()

    @classmethod
    def empty_cache(cls) -> None:
        """Release cached GPU memory back to system.

        Call this after freeing large tensors to ensure VRAM is released.
        Only effective when MPS is available.
        """
        if cls.is_available():
            torch.mps.empty_cache()


__all__ = ['GPUContext']
