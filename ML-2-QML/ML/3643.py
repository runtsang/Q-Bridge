"""Hybrid kernel implementation combining classical RBF and a learnable linear kernel."""

from __future__ import annotations

import numpy as np
from typing import Sequence

import torch
from torch import nn

class RBFKernel(nn.Module):
    """Classical Radial Basis Function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF similarity between two batches."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class LinearKernel(nn.Module):
    """Learnable linear kernel implemented as a weight matrix."""
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute weighted dot product."""
        x_proj = self.proj(x)
        y_proj = self.proj(y)
        return torch.sum(x_proj * y_proj, dim=-1, keepdim=True)

class HybridKernel(nn.Module):
    """Hybrid kernel combining RBF and a learnable linear kernel."""
    def __init__(self, input_dim: int, gamma: float = 1.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.rbf = RBFKernel(gamma)
        self.linear = LinearKernel(input_dim)
        self.alpha = alpha  # weight for RBF component

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return weighted sum of RBF and linear kernels."""
        rbf_val = self.rbf(x, y)
        lin_val = self.linear(x, y)
        return self.alpha * rbf_val + (1.0 - self.alpha) * lin_val

def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    input_dim: int,
    gamma: float = 1.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """Compute Gram matrix for two datasets using HybridKernel."""
    kernel = HybridKernel(input_dim, gamma, alpha)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["RBFKernel", "LinearKernel", "HybridKernel", "kernel_matrix"]
