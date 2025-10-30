"""Enhanced classical CNN‑FC model with residual fusion and richer output."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualFusionBlock(nn.Module):
    """Residual block that fuses two feature maps with a learnable gate."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.shortcut = nn.Sequential()
        if in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = (1 - self.gate) * identity + self.gate * out
        out = F.relu(out)
        return out

class QFCModelEnhanced(nn.Module):
    """Three‑layer CNN with residual fusion, followed by a fully‑connected head producing six features."""
    def __init__(self) -> None:
        super().__init__()
        # Conv backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualFusionBlock(8, 16),
            nn.MaxPool2d(2),
            ResidualFusionBlock(16, 32),
            nn.MaxPool2d(2),
        )
        # Flatten and FC head
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # richer output
        )
        self.norm = nn.BatchNorm1d(6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

__all__ = ["QFCModelEnhanced"]
