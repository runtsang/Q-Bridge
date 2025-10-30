"""Hybrid kernel combining classical RBF and CNN feature extraction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class RBFAnsatz(nn.Module):
    """Classical RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridQuantumKernel(nn.Module):
    """
    Hybrid kernel that optionally embeds data through a CNN before computing RBF.

    Parameters
    ----------
    gamma : float
        RBF width parameter.
    use_cnn : bool
        If True, a simple 2‑layer CNN extracts features before the kernel.
    """
    def __init__(self, gamma: float = 1.0, use_cnn: bool = True) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_cnn = use_cnn
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.rbf = RBFAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            x = self.cnn(x).view(x.shape[0], -1)
            y = self.cnn(y).view(y.shape[0], -1)
        return self.rbf(x, y)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0, use_cnn: bool = True) -> np.ndarray:
    """
    Compute the Gram matrix between two collections of samples.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of input tensors.
    gamma : float
        RBF width parameter.
    use_cnn : bool
        Whether to apply the CNN feature extractor.

    Returns
    -------
    np.ndarray
        The Gram matrix of shape (len(a), len(b)).
    """
    kernel = HybridQuantumKernel(gamma, use_cnn)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["HybridQuantumKernel", "kernel_matrix"]
