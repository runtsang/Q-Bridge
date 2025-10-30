"""Hybrid classical model combining CNN, classical quanvolution filter, and fully connected head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridNATModel(nn.Module):
    """Classical implementation of the Quantum‑NAT architecture with an added quanvolution‑style filter.

    The model first extracts convolutional features, then applies a 2×2 quanvolution‑style filter
    to capture local correlations, and finally projects the flattened representation through a
    two‑layer fully connected head.  The output dimensionality matches the original QFCModel
    (four features) and is batch‑normalised for stable training.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: two convolution + pooling stages
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical quanvolution filter implemented as a 2×2 conv with 4 output channels
        self.qfilter = nn.Conv2d(16, 4, kernel_size=2, stride=2, bias=False)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # Extract CNN features
        feats = self.features(x)
        # Apply quanvolution filter
        qfeat = self.qfilter(feats)
        # Flatten and classify
        flat = qfeat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


__all__ = ["HybridNATModel"]
