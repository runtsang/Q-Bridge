"""Hybrid classical kernel model with convolutional feature extractor."""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

# --------------------- Classical Kernel --------------------- #
class RBFKernel(nn.Module):
    """Radial basis function kernel with trainable gamma."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (N, D) or (D,)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, D)
        sq = torch.sum(diff * diff, dim=-1)      # (N, M)
        return torch.exp(-self.gamma * sq)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------- CNN Feature Extractor --------------------- #
class ConvFeatureExtractor(nn.Module):
    """Light‑weight CNN that produces a 4‑dimensional embedding."""
    def __init__(self):
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
        feats = self.features(x)
        flattened = feats.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

# --------------------- Hybrid Model --------------------- #
class HybridQuantumKernelModel(nn.Module):
    """
    Combines a classical CNN + RBF kernel with a quantum kernel module
    (defined separately in the quantum branch). The forward pass returns
    a tuple: (classical_embedding, kernel_features, quantum_embedding).
    """
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.cnn = ConvFeatureExtractor()
        self.kernel = RBFKernel(gamma)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Classical branch
        classical_emb = self.cnn(x)                      # (B, 4)
        # Kernel features between batch and itself
        kernel_feat = torch.zeros(x.shape[0], x.shape[0], device=x.device)
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                kernel_feat[i, j] = self.kernel(x[i], x[j])
        # Quantum branch placeholder (to be replaced by QKernel module)
        quantum_emb = torch.zeros_like(classical_emb)
        return classical_emb, kernel_feat, quantum_emb

__all__ = ["RBFKernel", "kernel_matrix", "ConvFeatureExtractor", "HybridQuantumKernelModel"]
