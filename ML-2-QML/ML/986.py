"""Hybrid classical kernel module with learnable gamma and batched support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

__all__ = ["QuantumKernelMethod", "kernel_matrix"]

class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel with a learnable gamma parameter.
    Supports batched inputs and automatic differentiation.
    """
    def __init__(self, gamma: float = 1.0, learnable_gamma: bool = False) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32), requires_grad=learnable_gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches of vectors.
        x: (B, D)
        y: (C, D)
        Returns: (B, C) kernel matrix.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        diff = x[:, None, :] - y[None, :, :]  # (B, C, D)
        sq_dist = torch.sum(diff * diff, dim=-1)  # (B, C)
        return torch.exp(-self.gamma * sq_dist)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """
    Compute Gram matrix between two sequences of tensors using a fixed gamma.
    """
    kernel = QuantumKernelMethod(gamma=gamma, learnable_gamma=False)
    return np.array([[kernel(x, y).item() for y in b] for x in a])
