"""Hybrid classical classifier that blends CNN features with a quantum‑inspired Random Fourier mapping."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class RandomFourierFeatures(nn.Module):
    """Random Fourier feature mapping to emulate a quantum kernel."""

    def __init__(self, input_dim: int, output_dim: int, sigma: float = 1.0) -> None:
        super().__init__()
        # Fixed random projection matrix and bias
        self.W = nn.Parameter(
            torch.randn(input_dim, output_dim) * sigma, requires_grad=False
        )
        self.b = nn.Parameter(
            2 * math.pi * torch.rand(output_dim), requires_grad=False
        )
        self.scale = math.sqrt(2.0 / output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.W + self.b
        return self.scale * torch.cos(proj)


class QuantumHybridClassifier(nn.Module):
    """Classical hybrid model inspired by Quantum‑NAT and incremental classifier.

    The network first extracts convolutional features, then projects them into a
    high‑dimensional Random Fourier feature space before passing through a
    depth‑parameterised fully‑connected head.  The final layer is batch‑norm‑
    stabilized to aid training stability.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        depth: int = 2,
        rff_dim: int = 64,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Determine flattened feature dimension
        dummy = torch.zeros(1, in_channels, 28, 28)
        feat_dim = self.features(dummy).view(1, -1).size(1)

        self.rff = RandomFourierFeatures(feat_dim, rff_dim)

        # Classifier head with configurable depth
        layers = []
        in_dim = rff_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, 64))
            layers.append(nn.ReLU(inplace=True))
            in_dim = 64
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flattened = feats.view(bsz, -1)
        rff_out = self.rff(flattened)
        logits = self.classifier(rff_out)
        return self.norm(logits)


__all__ = ["QuantumHybridClassifier", "RandomFourierFeatures"]
