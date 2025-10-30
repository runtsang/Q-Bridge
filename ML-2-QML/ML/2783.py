"""Hybrid kernel combining classical QCNN feature extraction with RBF kernel."""

from __future__ import annotations

import numpy as np
from typing import Sequence

import torch
from torch import nn


class QCNNFeatureExtractor(nn.Module):
    """Classical QCNN‑inspired feature extractor mirroring the quantum architecture."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x


class RBFKernel(nn.Module):
    """Standard radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridKernel(nn.Module):
    """Hybrid kernel that first extracts QCNN features then applies an RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.feature_extractor = QCNNFeatureExtractor()
        self.rbf = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fx = self.feature_extractor(x).unsqueeze(0)
        fy = self.feature_extractor(y).unsqueeze(0)
        return self.rbf(fx, fy).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of inputs using the hybrid kernel."""
    kernel = HybridKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["QCNNFeatureExtractor", "RBFKernel", "HybridKernel", "kernel_matrix"]
