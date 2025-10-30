"""Robust RBF kernel implementation for classical machine learning pipelines."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

class RBFKernel(nn.Module):
    """Classical Radial Basis Function kernel with efficient batch support."""
    def __init__(self, gamma: float = 1.0, batch_size: int = 1024) -> None:
        super().__init__()
        self.gamma = gamma
        self.batch_size = batch_size

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are 2‑D: (N, D) and (M, D)
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        # Compute pairwise squared Euclidean distances
        sq_norm_x = (x**2).sum(dim=1, keepdim=True)
        sq_norm_y = (y**2).sum(dim=1, keepdim=True).t()
        distances = sq_norm_x - 2 * x @ y.t() + sq_norm_y
        return torch.exp(-self.gamma * distances)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of feature vectors."""
    kernel = RBFKernel(gamma=gamma)
    # Stack tensors into a single matrix for efficient computation
    A = torch.vstack([t.reshape(-1, t.shape[-1]) for t in a])
    B = torch.vstack([t.reshape(-1, t.shape[-1]) for t in b])
    return kernel(A, B).cpu().numpy()

__all__ = ["RBFKernel"]
