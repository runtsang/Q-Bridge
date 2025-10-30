"""Hybrid classical kernel and neural network model.

This module combines a classical RBF kernel with a CNN‑based feature extractor,
providing a two‑stage pipeline: first compute similarity to a small support set,
then classify with a fully‑connected head.  The design mirrors the structure
presented in the original seed while adding a lightweight kernel
layer that can be swapped for a quantum variant.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Sequence


class RBFKernelAnsatz(nn.Module):
    """Classical RBF kernel computation."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class RBFKernel(nn.Module):
    """Wrapper around :class:`RBFKernelAnsatz` that normalises inputs."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = RBFKernelAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Expect 1‑D feature vectors.  Broadcast to (1, -1) for pairwise use.
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two datasets."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class ClassicalQFCModel(nn.Module):
    """CNN followed by a fully connected head, matching the seed."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


class HybridKernelNATModel(nn.Module):
    """Hybrid model that first maps inputs through a kernel then classifies."""
    def __init__(self,
                 kernel: nn.Module,
                 support: torch.Tensor,
                 model: nn.Module) -> None:
        """
        Args:
            kernel: Kernel module that returns a similarity vector.
            support: 1‑D tensor of support points (size 4).
            model: Classifier that consumes the kernel output.
        """
        super().__init__()
        self.kernel = kernel
        self.support = support
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute similarity to support set.
        k = self.kernel(x, self.support)          # (batch, 4)
        # Flatten to (batch, 4) if necessary.
        k = k.view(x.shape[0], -1)
        return self.model(k)


__all__ = [
    "RBFKernelAnsatz",
    "RBFKernel",
    "kernel_matrix",
    "ClassicalQFCModel",
    "HybridKernelNATModel",
]
