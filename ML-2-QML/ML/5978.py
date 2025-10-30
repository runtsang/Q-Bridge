"""Hybrid classical model combining CNN feature extractor, RBF kernel embedding, and linear classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Sequence

class CNNFeatureExtractor(nn.Module):
    """CNN backbone mirroring the original QFCModel feature extractor."""
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

class RBFKernel(nn.Module):
    """Batch RBF kernel between two batches of vectors."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        diff = X.unsqueeze(1) - Y.unsqueeze(0)  # [n, m, d]
        dist2 = (diff * diff).sum(dim=-1)  # [n, m]
        return torch.exp(-self.gamma * dist2)

class HybridNAT(nn.Module):
    """Classical hybrid network that embeds CNN features through an RBF kernel."""
    def __init__(
        self,
        n_classes: int = 4,
        reference_vectors: Optional[torch.Tensor] = None,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.kernel = RBFKernel(gamma)
        self.reference_vectors = reference_vectors
        if reference_vectors is not None:
            self.n_ref = reference_vectors.shape[0]
            self.fc = nn.Linear(self.n_ref, n_classes)
        else:
            self.fc = nn.Linear(16 * 7 * 7, n_classes)
        self.norm = nn.BatchNorm1d(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.cnn(x)  # [bsz, 16, 7, 7]
        flat = features.view(bsz, -1)  # [bsz, 784]
        if self.reference_vectors is not None:
            k = self.kernel(flat, self.reference_vectors)  # [bsz, n_ref]
            logits = self.fc(k)
        else:
            logits = self.fc(flat)
        return self.norm(logits)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Convenience wrapper returning a NumPy Gram matrix for arbitrary sequences."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridNAT", "kernel_matrix"]
