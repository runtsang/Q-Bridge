"""Hybrid Nat Model: Classical CNN backbone for 4-class classification.

This module implements the classical side of the HybridNatModel, inspired by
the QuantumNAT and Quanvolution seeds.  It contains a lightweight
convolutional feature extractor followed by a deep fully‑connected head.
The design keeps the network small enough for rapid prototyping while still
providing a rich feature representation that can be fused with a quantum
encoder in a separate module.

The architecture:
    Conv1 -> ReLU -> MaxPool
    Conv2 -> ReLU -> MaxPool
    Flatten -> FC1 -> ReLU -> FC2 -> ReLU -> FC3 (4 outputs)
    BatchNorm1d
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridNatModel(nn.Module):
    """Classical CNN + FC head for 4‑class classification."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


__all__ = ["HybridNatModel"]
