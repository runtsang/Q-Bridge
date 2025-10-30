"""Hybrid kernel combining RBF, convolutional feature extraction, and quantum-inspired design.

This module retains the original `KernalAnsatz` and `Kernel` classes for backward compatibility,
while adding a `ConvFilter` for classical convolutional feature extraction and a new
`HybridKernelMethod` that multiplies the RBF similarity with the convolutional similarity.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """Radial basis function (RBF) kernel implemented in PyTorch.

    Parameters
    ----------
    gamma : float, default 1.0
        Width of the Gaussian kernel.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Convenient wrapper around :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix for two datasets using the RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# Classical convolutional filter, inspired by the `Conv` class in the seed
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """2‑D convolutional filter that emulates a quantum quanvolution layer.

    Attributes
    ----------
    kernel_size : int
        Size of the square filter.
    threshold : float
        Activation threshold applied after the convolution.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        """Apply the filter to a 2‑D array and return the mean activation."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


def Conv() -> ConvFilter:
    """Factory returning a `ConvFilter` instance."""
    return ConvFilter()


# --------------------------------------------------------------------------- #
# Hybrid kernel that multiplies RBF similarity with convolutional similarity
# --------------------------------------------------------------------------- #
class HybridKernelMethod(nn.Module):
    """Hybrid similarity combining RBF and convolutional feature similarity.

    Parameters
    ----------
    gamma : float, default 1.0
        Width parameter for the RBF kernel.
    kernel_size : int, default 2
        Size of the convolutional filter.
    threshold : float, default 0.0
        Threshold used in the convolutional activation.
    """
    def __init__(self, gamma: float = 1.0, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.rbf = KernalAnsatz(gamma)
        self.conv = ConvFilter(kernel_size, threshold)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the product of RBF and convolutional similarities."""
        rbf_val = self.rbf(x, y)
        conv_x = self.conv.run(x.numpy())
        conv_y = self.conv.run(y.numpy())
        conv_val = conv_x * conv_y
        return rbf_val * conv_val

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the hybrid Gram matrix for two datasets."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "ConvFilter",
    "Conv",
    "HybridKernelMethod",
]
