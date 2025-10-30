"""Hybrid classical kernel combining RBF and QCNN feature extraction."""
from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Sequence

class QCNNFeatureExtractor(nn.Module):
    """Classical feature extractor mimicking a QCNN architecture."""
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x  # raw feature vector

class HybridKernel(nn.Module):
    """Hybrid kernel module that can operate in classical RBF mode or augment with QCNN features."""
    def __init__(self, gamma: float = 1.0, use_qcnn: bool = False, input_dim: int = 8) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_qcnn = use_qcnn
        self.qcnn = QCNNFeatureExtractor(input_dim) if use_qcnn else None

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_qcnn:
            x = self.qcnn(x)
            y = self.qcnn(y)
        return self._rbf(x, y).squeeze()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self(x, y).item() for y in b] for x in a])

__all__ = ["QCNNFeatureExtractor", "HybridKernel"]
