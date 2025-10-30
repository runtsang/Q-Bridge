"""Hybrid kernel for classical RBF and batch evaluation with optional shot noise."""

from __future__ import annotations

from typing import Sequence, Iterable, List

import numpy as np
import torch
from torch import nn

class HybridKernel(nn.Module):
    """Classic radial‑basis function kernel with optional quantum weighting."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a single kernel value for two input vectors."""
        x = x.unsqueeze(0) if x.ndim == 1 else x
        y = y.unsqueeze(0) if y.ndim == 1 else y
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff ** 2, dim=-1, keepdim=True)).squeeze()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two collections of vectors."""
        return np.array([[self(x, y).item() for y in b] for x in a])

class FastHybridEstimator:
    """Batch evaluation of a kernel with optional Gaussian shot noise."""
    def __init__(self, kernel: nn.Module) -> None:
        self.kernel = kernel

    def evaluate(
        self,
        X: Sequence[torch.Tensor],
        Y: Sequence[torch.Tensor],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        mat = self.kernel.kernel_matrix(X, Y)
        if shots is None:
            return mat
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1 / np.sqrt(shots), mat.shape)
        return mat + noise

__all__ = ["HybridKernel", "FastHybridEstimator"]
