"""Hybrid classical model combining CNN, RBF kernel, and fully connected layers."""

from __future__ import annotations

import torch
import torch.nn as nn

class HybridNATModel(nn.Module):
    """CNN + RBF kernel + FC for 4‑class classification."""
    def __init__(self, num_prototypes: int = 8, gamma: float = 1.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Prototypes in feature space
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 16 * 7 * 7))
        self.gamma = gamma
        self.fc = nn.Sequential(
            nn.Linear(num_prototypes, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        # RBF kernel between flat and prototypes
        diff = flat.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (bsz, num_proto, feat)
        dist_sq = torch.sum(diff * diff, dim=2)
        k = torch.exp(-self.gamma * dist_sq)
        out = self.fc(k)
        return self.norm(out)

__all__ = ["HybridNATModel"]
