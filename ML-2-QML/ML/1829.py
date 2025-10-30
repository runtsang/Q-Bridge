"""Extended classical RBF kernel with automatic gamma selection and batched GPU support.

The module defines a class QuantumKernelMethod that can compute the classical RBF kernel
in a batched manner on GPU and automatically selects the bandwidth if not supplied.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, Optional

class RBFKernel(nn.Module):
    """Radial Basis Function kernel with optional automatic bandwidth selection."""
    def __init__(self, gamma: Optional[float] = None):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x - y||^2)."""
        diff = x - y
        dist_sq = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * dist_sq)

def _median_pairwise_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute median of pairwise Euclidean distances between two batches."""
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    dist_sq = torch.sum(diff * diff, dim=-1)
    dist = torch.sqrt(dist_sq + 1e-12)
    return torch.median(dist).item()

class QuantumKernelMethod:
    """Hybrid interface that currently implements only the classical RBF kernel."""
    def __init__(self, gamma: Optional[float] = None, batch_size: int = 1024):
        self.gamma = gamma
        self.batch_size = batch_size
        self.kernel = RBFKernel(gamma)

    def _ensure_gamma(self, a: torch.Tensor, b: torch.Tensor):
        """If gamma is None compute a bandwidth based on the median heuristic."""
        if self.gamma is None:
            median = _median_pairwise_distance(a, b)
            self.gamma = 1.0 / (2 * median * median + 1e-12)
            self.kernel.gamma = self.gamma

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel value for two vectors."""
        self._ensure_gamma(x, y)
        return self.kernel(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix in batches, optionally on GPU."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a = torch.stack([x.to(device) for x in a])
        b = torch.stack([x.to(device) for x in b])
        self._ensure_gamma(a, b)
        n_a, n_b = a.shape[0], b.shape[0]
        K = torch.empty((n_a, n_b), device=device)
        for i in range(0, n_a, self.batch_size):
            a_batch = a[i:i+self.batch_size]
            for j in range(0, n_b, self.batch_size):
                b_batch = b[j:j+self.batch_size]
                diff = a_batch.unsqueeze(1) - b_batch.unsqueeze(0)
                dist_sq = torch.sum(diff * diff, dim=-1)
                K[i:i+self.batch_size, j:j+self.batch_size] = torch.exp(-self.gamma * dist_sq)
        return K.cpu().numpy()

    def kernel_matrix_multibatch(self, datasets: Iterable[Sequence[torch.Tensor]]) -> np.ndarray:
        """Compute a block‑wise Gram matrix for a list of datasets."""
        all_tensors = [torch.stack(ds) for ds in datasets]
        combined = torch.cat(all_tensors, dim=0)
        combined_list = [combined[i] for i in range(combined.shape[0])]
        return self.kernel_matrix(combined_list, combined_list)

__all__ = ["QuantumKernelMethod"]
