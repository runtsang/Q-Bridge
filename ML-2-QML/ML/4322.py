"""Hybrid classical classifier combining convolution, pooling, and fully connected layers."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class HybridQuantumClassifier(nn.Module):
    """
    A hybrid classical classifier that emulates quantum-inspired layers.
    It stacks 2D convolutions, pooling, and a fully connected head,
    mirroring the structure of QCNN and QuantumNAT models.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        conv_channels: list[int] = [8, 16],
        pool_size: int = 2,
        fc_layers: list[int] = [64, 32],
    ) -> None:
        super().__init__()
        # Feature extraction block
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(conv_channels[1], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.flatten = nn.Flatten()
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[1], fc_layers[0]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_layers[0], fc_layers[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_layers[1], num_classes),
        )
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.bn(x)
        return torch.sigmoid(x) if self.fc[-1].out_features == 1 else F.softmax(x, dim=1)

__all__ = ["HybridQuantumClassifier"]
