"""Hybrid classical CNN for the QuantumNAT task.

The model extends the original QFCModel by exposing a configurable
convolutional backbone followed by a fully‑connected projection.
It is deliberately kept lightweight so that it can serve as a
drop‑in replacement for the quantum version in ablation studies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the classical Conv filter if desired
# from.Conv import Conv

class HybridNATModel(nn.Module):
    """Classical CNN with optional ConvFilter integration."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Compute feature size after pooling for a 28x28 input
        dummy = torch.zeros(1, in_channels, 28, 28)
        feat_size = self.features(dummy).view(1, -1).size(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

        # Optional: replace the pooling layer with the classical Conv filter
        # self.conv_filter = Conv()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.classifier(flat)
        return self.norm(out)

__all__ = ["HybridNATModel"]
