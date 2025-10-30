"""Hybrid classical kernel module that marries CNN feature extraction with an RBF kernel."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    """Extracts a 64‑dimensional embedding from 1‑channel images."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 28x28 -> 14x14
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 14x14 -> 7x7
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 7 * 7, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        return self.fc(x)

class HybridKernelMethod(nn.Module):
    """Classical kernel that applies a CNN feature extractor followed by an RBF kernel."""
    def __init__(self, gamma: float = 1.0, extractor: nn.Module | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.extractor = extractor if extractor is not None else CNNFeatureExtractor()

    def _rbf(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel matrix between batches `x` and `y`."""
        f_x = self.extractor(x)
        f_y = self.extractor(y)
        return self._rbf(f_x, f_y)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Utility that builds a Gram matrix for arbitrary tensors using the hybrid kernel."""
    model = HybridKernelMethod(gamma)
    return np.array([[model(a_i, b_j).item() for b_j in b] for a_i in a])

__all__ = ["HybridKernelMethod", "kernel_matrix"]
