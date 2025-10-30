"""Hybrid kernel method combining classical RBF and quantum‑inspired convolution.

The module exposes a single :class:`HybridKernelMethod` class that can compute
classical RBF kernels, optionally pre‑process data with a 2‑D convolutional
filter, and serve as a drop‑in replacement for the legacy ``QuantumKernelMethod``.
"""

import numpy as np
import torch
from torch import nn
from typing import List, Sequence


class ConvFilter(nn.Module):
    """Lightweight 2‑D convolutional filter that mimics a quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution, sigmoid non‑linearity, and return the mean activation."""
        x = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def run(self, data) -> float:
        """Convenience wrapper that accepts a numpy array and returns a Python float."""
        return self.forward(torch.as_tensor(data, dtype=torch.float32)).item()


class HybridKernelMethod:
    """Hybrid kernel class that supports classical RBF kernels and an optional
    convolutional pre‑processor.  The class is compatible with the legacy
    ``QuantumKernelMethod`` API, making it a drop‑in replacement for existing
    workflows."""
    def __init__(self,
                 gamma: float = 1.0,
                 use_convolution: bool = False,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0) -> None:
        self.gamma = gamma
        self.use_convolution = use_convolution
        self.conv_filter = ConvFilter(conv_kernel_size, conv_threshold) if use_convolution else None

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel between two vectors."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute a Gram matrix between two collections of vectors.  If
        ``use_convolution`` is True, each vector is first passed through the
        convolutional filter."""
        if self.use_convolution:
            a = [torch.tensor(self.conv_filter.run(v.numpy()), dtype=torch.float32) for v in a]
            b = [torch.tensor(self.conv_filter.run(v.numpy()), dtype=torch.float32) for v in b]

        kernel_vals = []
        for x in a:
            row = [self._rbf(x, y).item() for y in b]
            kernel_vals.append(row)
        return np.array(kernel_vals)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Alias to :meth:`_rbf` for compatibility with the original API."""
        return self._rbf(x, y)


__all__ = ["HybridKernelMethod"]
