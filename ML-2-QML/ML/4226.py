import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class RBFKernel(nn.Module):
    """Classical RBF kernel implemented in PyTorch."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that exposes the same API as the original QuantumKernelMethod."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two sequences of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolution that emulates the original quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class EstimatorNN(nn.Module):
    """Light‑weight regression network that mirrors EstimatorQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * 14 * 14, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class HybridKernelEstimator(nn.Module):
    """
    Classical hybrid model that brings together:
    * an RBF kernel for vector data,
    * a quanvolution filter for image feature extraction,
    * and a simple regression network.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.kernel = Kernel(gamma)
        self.qfilter = QuanvolutionFilter()
        self.estimator = EstimatorNN()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the kernel Gram matrix between two datasets."""
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image data.
        Assumes ``x`` is a batch of 1×28×28 grayscale images.
        """
        features = self.qfilter(x)
        return self.estimator(features)

__all__ = [
    "RBFKernel",
    "Kernel",
    "kernel_matrix",
    "QuanvolutionFilter",
    "EstimatorNN",
    "HybridKernelEstimator",
]
