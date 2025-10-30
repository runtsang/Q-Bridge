"""Hybrid kernel method combining classical RBF kernel with CNN feature extraction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Sequence


class CNNFeatureExtractor(nn.Module):
    """Simple CNN that maps 1‑channel images to a 4‑dimensional feature vector."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RBFKernel(nn.Module):
    """Classical RBF kernel operating on feature vectors."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridKernelMethod(nn.Module):
    """
    Hybrid kernel that first extracts features via a CNN and then applies an RBF kernel.
    The class can be used as a drop‑in replacement for a conventional kernel module.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor()
        self.kernel = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel value between two batches of images."""
        x_feat = self.feature_extractor(x)
        y_feat = self.feature_extractor(y)
        return self.kernel(x_feat, y_feat)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix for two sets of images."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])


__all__ = ["HybridKernelMethod"]
