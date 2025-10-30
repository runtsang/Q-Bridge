"""
Hybrid classical model combining a CNN, RBF kernel, and prototype similarity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RBFKernel(nn.Module):
    """Gaussian radial basis kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (N, D), y: (M, D) -> (N, M)
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-self.gamma * (diff.pow(2).sum(-1)))


class HybridNATModel(nn.Module):
    """CNN + RBF kernel + prototype classifier."""

    def __init__(self, num_prototypes: int = 4, gamma: float = 1.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Infer feature dimensionality
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            dummy_feat = self.features(dummy)
        feat_dim = dummy_feat.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_prototypes),
        )
        self.kernel = RBFKernel(gamma)
        # Learnable prototype vectors in feature space
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feat_dim))
        self.norm = nn.BatchNorm1d(num_prototypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x).view(bsz, -1)
        logits_fc = self.fc(feat)
        logits_kernel = self.kernel(feat, self.prototypes)
        logits = logits_fc + logits_kernel
        return self.norm(logits)


__all__ = ["HybridNATModel"]
