"""Enhanced classical RBF kernel with caching and vectorised gram matrix."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Tuple

__all__ = ["QuantumKernelMethod", "kernel_matrix"]


class QuantumKernelMethod(nn.Module):
    """Classical RBF kernel with optional caching.

    Parameters
    ----------
    gamma : float, default=1.0
        Width of the Gaussian kernel.
    cache_size : int, default=1024
        Number of evaluations to keep in an LRU‑style cache.
    """

    def __init__(self, gamma: float = 1.0, cache_size: int = 1024) -> None:
        super().__init__()
        self.gamma = gamma
        self.cache_size = cache_size
        self._cache: dict[Tuple[int, int], float] = {}

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the RBF kernel between two 1‑D tensors."""
        key = (id(x), id(y))
        if key in self._cache:
            return torch.tensor(self._cache[key], device=x.device, dtype=x.dtype)
        diff = x - y
        val = torch.exp(-self.gamma * torch.sum(diff * diff))
        if len(self._cache) < self.cache_size:
            self._cache[key] = val.item()
        return val

    @torch.no_grad()
    def gram_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Return the Gram matrix between two batches of vectors."""
        a_stack = torch.stack(a)
        b_stack = torch.stack(b)
        diff = a_stack.unsqueeze(1) - b_stack.unsqueeze(0)  # (len(a), len(b), d)
        sq_norm = torch.sum(diff * diff, dim=-1)
        kernel = torch.exp(-self.gamma * sq_norm)
        return kernel.numpy()


def kernel_matrix(
    a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0
) -> np.ndarray:
    """Convenience wrapper that constructs a cached kernel."""
    model = QuantumKernelMethod(gamma)
    return model.gram_matrix(a, b)
