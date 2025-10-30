"""Classical counterpart of the QuantumNATHybrid model.

The module mirrors the quantum architecture while remaining purely classical.
It is compatible with the anchor file QuantumNAT.py but expands the network
with residual blocks, dropout, and a hybrid sigmoid head inspired by the
HybridFunction in the quantum module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Small residual block used in the CNN backbone."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out += self.shortcut(x)
        return self.relu(out)


class HybridSigmoid(nn.Module):
    """Sigmoid head with an optional shift, mimicking the quantum expectation."""

    def __init__(self, in_features: int, shift: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=bias)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)


class QuantumNATHybrid(nn.Module):
    """CNN‑based model that mirrors the quantum hybrid architecture."""

    def __init__(self) -> None:
        super().__init__()
        # Backbone
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.res1 = ResidualBlock(8, 16, stride=2)
        self.res2 = ResidualBlock(16, 32, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected head
        self.fc1 = nn.Linear(32, 64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 4)
        self.norm = nn.BatchNorm1d(4)

        # Hybrid sigmoid head
        self.hybrid_head = HybridSigmoid(4, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        # FC head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x)

        # Hybrid sigmoid (optional)
        probs = self.hybrid_head(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumNATHybrid"]
