"""Enhanced classical binary classifier with residual blocks and a hybrid sigmoid head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with Conv‑BN‑ReLU‑Conv‑BN and skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class Hybrid(nn.Module):
    """Hybrid sigmoid head that emulates the quantum expectation."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x) + self.shift
        return torch.sigmoid(logits)


class QCNet(nn.Module):
    """Classical CNN with residual blocks and hybrid sigmoid head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = ResidualBlock(32)
        self.res2 = ResidualBlock(32)
        self.drop = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 64)
        self.hybrid = Hybrid(64, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prob = self.hybrid(x)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["ResidualBlock", "Hybrid", "QCNet"]
