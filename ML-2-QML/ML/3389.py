"""Hybrid convolutional filter with optional RBF kernel, classical implementation."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Sequence

class HybridConvKernel(nn.Module):
    """Combines a 2‑D convolution filter with an RBF kernel for feature extraction.

    The class can operate purely classically or emulate the quantum filter via a
    classical surrogate.  It exposes a ``run`` method that accepts a 2‑D array
    and returns a scalar feature.  When ``use_kernel`` is True, the output is
    augmented with an inner product against a pre‑computed kernel matrix.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 use_kernel: bool = False,
                 gamma: float = 1.0,
                 kernel_data: Sequence[np.ndarray] | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_kernel = use_kernel
        self.gamma = gamma
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Pre‑compute kernel matrix if data provided
        self.kernel_matrix = None
        if self.use_kernel and kernel_data is not None:
            tensors = [torch.tensor(x, dtype=torch.float32).view(1, -1) for x in kernel_data]
            self.kernel_matrix = np.array([[self._rbf(x, y).item() for y in tensors] for x in tensors])

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def run(self, data: np.ndarray) -> float:
        """Apply the classical filter and optionally a kernel."""
        tensor = torch.as_tensor(data, dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        feature = activations.mean().item()
        if self.use_kernel and self.kernel_matrix is not None:
            vec = torch.tensor(data, dtype=torch.float32).view(1, -1)
            sims = []
            for row in self.kernel_matrix:
                sims.append(np.exp(-self.gamma * np.sum((vec - row)**2)))
            return feature + np.mean(sims)
        return feature

def Conv() -> HybridConvKernel:
    """Return a drop‑in replacement for the quantum filter."""
    return HybridConvKernel()

__all__ = ["HybridConvKernel", "Conv"]
