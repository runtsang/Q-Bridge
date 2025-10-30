"""Hybrid classical kernel module with feature extraction and RBF kernel."""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

class FeatureExtractor(nn.Module):
    """CNN feature extractor inspired by QuantumNAT."""
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
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

class RBFKernel(nn.Module):
    """Classical RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridKernel(nn.Module):
    """Weighted blend of classical RBF and a user‑supplied quantum kernel."""
    def __init__(self,
                 gamma: float = 1.0,
                 alpha: float = 0.5,
                 quantum_module: nn.Module | None = None) -> None:
        super().__init__()
        self.rbf = RBFKernel(gamma)
        self.alpha = alpha
        self.quantum_module = quantum_module

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rbf_val = self.rbf(x, y)
        if self.quantum_module is not None:
            q_val = self.quantum_module(x, y)
            return self.alpha * q_val + (1.0 - self.alpha) * rbf_val
        return rbf_val

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """Compute classical RBF Gram matrix."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

def hybrid_kernel_matrix(a: Sequence[torch.Tensor],
                         b: Sequence[torch.Tensor],
                         gamma: float = 1.0,
                         alpha: float = 0.5,
                         quantum_module: nn.Module | None = None) -> np.ndarray:
    """Compute weighted hybrid Gram matrix."""
    c = kernel_matrix(a, b, gamma)
    if quantum_module is None:
        return c
    q = np.array([[quantum_module(x, y).item() for y in b] for x in a])
    return alpha * q + (1.0 - alpha) * c

__all__ = [
    "FeatureExtractor",
    "RBFKernel",
    "HybridKernel",
    "kernel_matrix",
    "hybrid_kernel_matrix",
]
