"""Enhanced classical CNN with residual connections and dropout for the Quantum‑NAT task."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATModel(nn.Module):
    """Classical CNN with residual blocks, dropout and multi‑head output for Quantum‑NAT."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        logits = self.classifier(flattened)
        return self.norm(logits)


class ResidualBlock(nn.Module):
    """Simple residual block with two conv layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


__all__ = ["QuantumNATModel"]
