import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class QuanvolutionFilter(nn.Module):
    """
    Classical 2x2 convolution filter inspired by the quanvolution example.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuantumKernelMethod(nn.Module):
    """
    Hybrid classical kernel method that optionally applies a classical quanvolution
    filter before computing the radial basis function kernel. The kernel is defined
    as K(x, y) = exp(-gamma * ||x - y||^2).
    """
    def __init__(self, gamma: float = 1.0, use_quanvolution: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_quanvolution = use_quanvolution
        if use_quanvolution:
            self.quanvolution = QuanvolutionFilter()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            x = self.quanvolution(x)
            y = self.quanvolution(y)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def apply_quanvolution(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_quanvolution:
            raise RuntimeError("Quanvolution filter not enabled.")
        return self.quanvolution(x)

__all__ = ["QuantumKernelMethod", "QuanvolutionFilter"]
