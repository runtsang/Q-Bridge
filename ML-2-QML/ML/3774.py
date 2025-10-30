"""Hybrid sampler network combining convolutional feature extraction and classical MLP."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerQNN(nn.Module):
    """Classical hybrid sampler network.

    Combines a CNN feature extractor with a fully‑connected sampler
    that outputs a 4‑class probability distribution.  This module
    can be used as a plug‑in for reinforcement‑learning agents or
    probabilistic generative models.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: mimic the 2‑layer CNN from QuantumNAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten to 16*7*7 = 784 features for 28x28 input
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return a probability distribution over 4 actions."""
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        out = self.norm(out)
        return F.softmax(out, dim=-1)


__all__ = ["HybridSamplerQNN"]
