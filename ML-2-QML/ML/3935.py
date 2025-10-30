"""QuantumHybridNAT: classical backbone + fully‑connected head.

This module defines the classical part of the hybrid model. It can be used
independently or as a drop‑in replacement for the quantum head in the
qml version.  The architecture is a lightweight convolutional feature
extractor followed by a fully‑connected head that maps to four
output logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumHybridNAT(nn.Module):
    """
    Classical backbone + fully‑connected head.  The forward method returns
    the output of a BatchNorm1d applied to the 4‑dimensional logits.
    """
    def __init__(self, in_channels: int = 1, n_filt: int = 8, n_features: int = 64) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, n_filt, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(n_filt, n_filt * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Size after pooling depends on input, but we assume 28x28
        # 28x28 -> 14x14 -> 7x7
        self.n_flat = n_filt * 2 * 7 * 7
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(self.n_flat, n_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_features, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        x = self.features(x)
        x = x.view(bsz, -1)
        x = self.fc(x)
        return self.norm(x)


__all__ = ["QuantumHybridNAT"]
