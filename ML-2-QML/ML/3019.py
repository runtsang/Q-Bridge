"""Hybrid classical estimator that fuses CNN feature extraction with a linear head.

The architecture mirrors the classical Quantum‑NAT CNN while adding a
trainable linear output that emulates a variational quantum circuit.
It can be used as a drop‑in replacement for the original EstimatorQNN
while providing a richer feature representation.
"""

from __future__ import annotations

import torch
from torch import nn


class EstimatorQNN(nn.Module):
    """Classical estimator with CNN feature extractor and linear head."""
    def __init__(self, feature_dim: int = 64, out_dim: int = 1) -> None:
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
            nn.Linear(16 * 7 * 7, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim),
        )
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


__all__ = ["EstimatorQNN"]
