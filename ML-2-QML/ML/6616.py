"""Hybrid classical kernel combining RBF and quanvolution features."""
from __future__ import annotations

import numpy as np
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalRBFKernel(nn.Module):
    """Standard RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class ClassicalQuanvolutionFilter(nn.Module):
    """2x2 patch-based convolution to produce features."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)  # (batch, 4, H/2, W/2)
        return features.view(x.size(0), -1)

class HybridKernel(nn.Module):
    """Hybrid kernel: apply quanvolution filter then RBF."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        self.rbf = ClassicalRBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        f_x = self.filter(x)
        f_y = self.filter(y)
        return self.rbf(f_x, f_y)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = HybridKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernel", "kernel_matrix"]
