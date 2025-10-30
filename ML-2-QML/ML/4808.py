"""Hybrid classical kernel model inspired by RBF kernels and fraud-detection scaling."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class HybridKernelAnsatz(nn.Module):
    """Trainable RBF kernel with optional normalization."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel value between two batches."""
        diff = x - y
        sqdist = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * sqdist)

class HybridKernelModel(nn.Module):
    """Map inputs to a kernel feature space and apply a linear head with scaling."""
    def __init__(self, support_vectors: torch.Tensor, out_dim: int = 1):
        super().__init__()
        self.support = support_vectors.clone().detach()
        self.ansatz = HybridKernelAnsatz()
        self.head = nn.Linear(self.support.size(0), out_dim)
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.exp(-self.ansatz.gamma * torch.cdist(x, self.support) ** 2)
        out = self.head(k)
        return self.scale * out + self.shift

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using a classical RBF kernel."""
    x = torch.stack(a)
    y = torch.stack(b)
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    sqdist = torch.sum(diff * diff, dim=-1)
    return torch.exp(-gamma * sqdist).cpu().numpy()

__all__ = ["HybridKernelModel", "kernel_matrix"]
