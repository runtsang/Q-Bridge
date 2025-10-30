"""Enhanced classical RBF kernel with learnable bandwidth and adaptive feature mapping."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BandwidthSelector(nn.Module):
    """Learnable bandwidth selector mapping a feature vector to a positive bandwidth."""
    def __init__(self, input_dim: int, hidden_size: int = 1, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size, bias=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map to positive value via softplus
        return F.softplus(self.linear(x)).squeeze(-1) + self.eps


class FeatureMapper(nn.Module):
    """Data‑adaptive feature map using a small neural network."""
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveRBFKernel(nn.Module):
    """RBF kernel with learnable bandwidth and optional feature mapping."""
    def __init__(self, input_dim: int, bandwidth: float = 1.0, use_mapper: bool = False):
        super().__init__()
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth, dtype=torch.float32))
        self.use_mapper = use_mapper
        if use_mapper:
            self.mapper = FeatureMapper(input_dim)
        else:
            self.mapper = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.mapper is not None:
            x = self.mapper(x)
            y = self.mapper(y)
        diff = x - y
        sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.bandwidth * sq_norm)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two sequences of tensors."""
        return np.array([[self(x, y).item() for y in b] for x in a])


__all__ = ["BandwidthSelector", "FeatureMapper", "AdaptiveRBFKernel"]
