"""Hybrid kernel combining CNN feature extraction and RBF kernel.

The class `Kernel` implements a lightweight convolutional network to
extract 4‑dimensional features from grayscale images.  These features
are then compared using an RBF kernel, providing a flexible
classical kernel that benefits from learned representations.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

class Kernel(nn.Module):
    """CNN + RBF kernel hybrid.

    Parameters
    ----------
    gamma : float, optional
        Width parameter for the RBF kernel.  Default is 1.0.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        # Feature extractor mirroring QFCModel from QuantumNAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Compute feature dimension for 28×28 input
        dummy = torch.zeros(1, 1, 28, 28)
        feat_dim = self.features(dummy).view(1, -1).size(1)
        # Projection to 4‑dimensional feature space
        self.fc = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return 4‑dimensional feature vector for each image."""
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        return self.fc(flat)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel between two batches of images."""
        fx = self.forward(x)
        fy = self.forward(y)
        diff = fx.unsqueeze(1) - fy.unsqueeze(0)
        dist2 = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * dist2)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using the hybrid kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel.kernel(x, y).item() for y in b] for x in a])

__all__ = ["Kernel", "kernel_matrix"]
