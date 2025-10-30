"""Hybrid classical RBF kernel with GPU and batched support.

The original `Kernel` class was a thin wrapper around an
`KernalAnsatz`.  Here we expose the kernel as a standalone
`HybridKernel` that accepts arbitrary torch tensors, can be
executed on GPU, and processes data in blocks to keep memory
footprint manageable.  The implementation is fully differentiable
and can be used inside larger neural‑network pipelines.

Key extensions:
* `device` argument to run on CUDA if available.
* `batch_size` controls the internal chunking of the Gram matrix.
* `kernel_matrix` now accepts either torch tensors or Python
  sequences, returning a NumPy array.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

class HybridKernel(nn.Module):
    """Classical RBF kernel with optional GPU support and batch processing."""
    def __init__(self, gamma: float = 1.0, device: str | torch.device = "cpu",
                 batch_size: int = 1024) -> None:
        super().__init__()
        self.gamma = gamma
        self.device = torch.device(device)
        self.batch_size = batch_size

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the RBF kernel value for two 1‑D tensors."""
        x = x.to(self.device)
        y = y.to(self.device)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: torch.Tensor | list[torch.Tensor],
                      b: torch.Tensor | list[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two collections of vectors."""
        if isinstance(a, list):
            a = torch.stack(a)
        if isinstance(b, list):
            b = torch.stack(b)
        a = a.to(self.device)
        b = b.to(self.device)

        n, d = a.shape
        m, _ = b.shape
        out = torch.empty((n, m), device=self.device)

        for i in range(0, n, self.batch_size):
            a_batch = a[i:i+self.batch_size]
            for j in range(0, m, self.batch_size):
                b_batch = b[j:j+self.batch_size]
                diff = a_batch.unsqueeze(1) - b_batch.unsqueeze(0)
                out[i:i+self.batch_size, j:j+self.batch_size] = torch.exp(
                    -self.gamma * torch.sum(diff * diff, dim=-1)
                )
        return out.cpu().numpy()

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor],
                  gamma: float = 1.0, device: str | torch.device = "cpu",
                  batch_size: int = 1024) -> np.ndarray:
    """Convenience wrapper that returns a NumPy array."""
    kernel = HybridKernel(gamma, device, batch_size)
    return kernel.kernel_matrix(a, b)

__all__ = ["HybridKernel", "kernel_matrix"]
