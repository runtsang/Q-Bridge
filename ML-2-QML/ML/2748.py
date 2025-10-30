from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class QuanvolutionFilter(nn.Module):
    """Classical quanvolution filter using a 2×2 convolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class HybridKernelAnsatz(nn.Module):
    """RBF kernel optionally preceded by a quanvolution feature extractor."""
    def __init__(self, gamma: float = 1.0, use_quanvolution: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_quanvolution = use_quanvolution
        if self.use_quanvolution:
            self.qfilter = QuanvolutionFilter()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            x = self.qfilter(x)
            y = self.qfilter(y)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wraps :class:`HybridKernelAnsatz`."""
    def __init__(self, gamma: float = 1.0, use_quanvolution: bool = False) -> None:
        super().__init__()
        self.ansatz = HybridKernelAnsatz(gamma, use_quanvolution)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0, use_quanvolution: bool = False) -> np.ndarray:
    kernel = Kernel(gamma, use_quanvolution)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernelAnsatz", "Kernel", "kernel_matrix"]
