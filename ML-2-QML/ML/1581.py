"""Enhanced classical architecture for QuantumNAT with residuals and dual‑head classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block used in the feature extractor."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class QuantumNATEnhanced(nn.Module):
    """Classical CNN with residual blocks and a dual‑head classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(8, 16, stride=1),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32, stride=1),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.binary_head = nn.Linear(256, 1)
        self.multi_head = nn.Linear(256, 4)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tuple (binary_logits, multi_logits)."""
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        binary = self.sigmoid(self.binary_head(x))
        multi = self.softmax(self.multi_head(x))
        return binary, multi

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature vector before the classifier heads."""
        x = self.features(x)
        return self.flatten(x)


__all__ = ["QuantumNATEnhanced"]
