"""GPU acceleration using PyTorch (CUDA or MPS backend)."""

import numpy as np
import torch
from typing import Optional
from torchvision.transforms.functional import gaussian_blur


class GPUContext:
    """Global GPU context for accelerated operations (CUDA or MPS)."""

    _device: Optional[torch.device] = None
    _available: Optional[bool] = None
    _backend: Optional[str] = None  # 'cuda', 'mps', or None

    @classmethod
    def device(cls) -> torch.device:
        """Get GPU device (lazy initialization).

        Returns CUDA if available, then MPS (Apple Silicon), otherwise CPU.
        PyTorch operations work on all devices, so CPU is a valid fallback.
        """
        if cls._device is None:
            try:
                if torch.cuda.is_available():
                    cls._device = torch.device('cuda')
                    cls._backend = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    cls._device = torch.device('mps')
                    cls._backend = 'mps'
                else:
                    cls._device = torch.device('cpu')
                    cls._backend = None
            except Exception:
                # Fallback to CPU on any GPU initialization error
                cls._device = torch.device('cpu')
                cls._backend = None
        return cls._device

    @classmethod
    def is_available(cls) -> bool:
        """Check if GPU acceleration is available (CUDA or MPS)."""
        if cls._available is None:
            try:
                cuda_ok = torch.cuda.is_available()
                mps_ok = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                cls._available = cuda_ok or mps_ok
            except Exception:
                cls._available = False
        return cls._available

    @classmethod
    def warmup(cls) -> None:
        """Pre-compile GPU kernels with dummy operations.

        Fails silently on errors to avoid blocking app startup.
        """
        if not cls.is_available():
            return

        try:
            device = cls.device()
            # Warmup with realistic operation sizes
            dummy = torch.randn(1024, 1024, device=device, dtype=torch.float32)

            # Gaussian blur warmup (torchvision will compile kernels)
            _ = gaussian_blur(dummy.unsqueeze(0).unsqueeze(0), kernel_size=5, sigma=2.0)

            # Percentile/quantile warmup
            _ = torch.quantile(dummy, torch.tensor([0.01, 0.99], device=device))

            # Arithmetic operations warmup
            _ = dummy * 1.5 + 0.5
            _ = torch.pow(dummy.clamp(0, 1), 1.2)

            # Synchronize to ensure all operations complete
            if cls._backend == 'cuda':
                torch.cuda.synchronize()
            elif cls._backend == 'mps' and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
        except Exception:
            # GPU warmup failed - fall back to CPU
            cls._device = torch.device('cpu')
            cls._backend = None
            cls._available = False

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
        """
        try:
            if cls._backend == 'cuda':
                torch.cuda.empty_cache()
            elif cls._backend == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        except Exception:
            pass  # Ignore cache clearing errors


__all__ = ['GPUContext']
