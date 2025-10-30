"""Hybrid classical model combining a depth‑separable CNN and a random
linear layer for 4‑class classification.  The design draws from the
original Quantum‑NAT seed but replaces the basic ConvNet with a
lighter, more regularised architecture and adds a stochastic
fully‑connected block to emulate quantum‑style randomness."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATGen(nn.Module):
    """Hybrid CNN + stochastic FC feature extractor for 4‑class output."""

    def __init__(self) -> None:
        super().__init__()
        # Depth‑separable conv block (efficient feature extractor)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Random fully‑connected layer to mimic quantum randomness
        self.random_fc = nn.Linear(32 * 7 * 7, 128)
        torch.nn.init.normal_(self.random_fc.weight, std=0.1)
        torch.nn.init.zeros_(self.random_fc.bias)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.BatchNorm1d(4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        rand_feat = F.relu(self.random_fc(flat))
        return self.classifier(rand_feat)


__all__ = ["QuantumNATGen"]
