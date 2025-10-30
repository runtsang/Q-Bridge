"""Hybrid classical kernel estimator with neural feature mapping.

This module combines a radial basis function kernel with a small
feed‑forward network, inspired by the EstimatorQNN architecture.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

class FeatureNet(nn.Module):
    """Simple feature extractor mimicking EstimatorQNN."""
    def __init__(self, input_dim: int, hidden_dim: int = 8, output_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RBFKernel(nn.Module):
    """Radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridKernelEstimator(nn.Module):
    """Hybrid kernel that maps data via a neural net then applies RBF kernel."""
    def __init__(self, input_dim: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.feature_net = FeatureNet(input_dim)
        self.kernel = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.feature_net(x)
        y = self.feature_net(y)
        return self.kernel(x, y).squeeze()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix for sequences of tensors."""
        mat = torch.stack([self.forward(x, y) for x in a for y in b])
        return mat.view(len(a), len(b)).numpy()

__all__ = ["HybridKernelEstimator", "FeatureNet", "RBFKernel"]
