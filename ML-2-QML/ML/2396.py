"""Hybrid classical RBF kernel with batch evaluation and optional Gaussian shot noise."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, List

class HybridKernel(nn.Module):
    """Classical RBF kernel supporting batch evaluation and Gaussian shot noise."""
    def __init__(self, gamma: float = 1.0, shots: int | None = None, seed: int | None = None):
        super().__init__()
        self.gamma = gamma
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def evaluate(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor], parameter_sets: Sequence[Sequence[float]]) -> List[np.ndarray]:
        """Return a list of kernel matrices, one per parameter set. Each parameter set is a list of scalars; the first scalar is interpreted as gamma."""
        results: List[np.ndarray] = []
        for params in parameter_sets:
            if params:
                self.gamma = float(params[0])
            K = self.kernel_matrix(X, Y)
            if self.shots is not None:
                noise = self.rng.normal(0, max(1e-6, 1.0 / self.shots), K.shape)
                K += noise
            results.append(K)
        return results

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = HybridKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernel", "kernel_matrix"]
