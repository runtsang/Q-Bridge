"""HybridQuantumNAT: Classical baseline combining CNN features with a lightweight regressor."""

from __future__ import annotations

import torch
from torch import nn


class HybridQuantumNAT(nn.Module):
    """
    Purely classical hybrid architecture.

    Architecture:
        - Two convolutional layers (1 → 8 → 16 channels) with ReLU and MaxPool.
        - Flatten + 2‑layer fully connected regressor (64 → 4 → 1) with Tanh activations.
        - Final output is a scalar regression value.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.regressor(flat)
        return self.norm(out)


__all__ = ["HybridQuantumNAT"]
