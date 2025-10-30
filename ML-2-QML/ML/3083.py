"""Hybrid classical model combining a CNN feature extractor and a lightweight regression head inspired by QuantumNAT and EstimatorQNN."""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridNATModel(nn.Module):
    """
    Classical hybrid model.

    This module extracts features with a shallow CNN and then
    regresses the target with a small fully‑connected network.
    The design is inspired by the classical part of Quantum‑NAT
    and the EstimatorQNN regressor.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor – same shape as original Quantum‑NAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Regression head – mirrors EstimatorQNN architecture
        self.regressor = nn.Sequential(
            nn.Linear(16 * 7 * 7, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        self.norm = nn.BatchNorm1d(16 * 7 * 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        normed = self.norm(flat)
        out = self.regressor(normed)
        return out


__all__ = ["HybridNATModel"]
