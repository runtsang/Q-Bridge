"""Enhanced classical binary classifier with residual blocks and a learnable activation.

This module extends the original hybrid classifier by incorporating
residual connections, batch normalization, and a learnable
shift that mimics the quantum head while remaining fully classical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with convolution, batch‑norm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn(self.conv(x)))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class HybridNet(nn.Module):
    """Classical CNN with a residual head and a learnable sigmoid‑like activation."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 64, stride=2),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        # Learnable shift for the sigmoid‑like activation
        self.shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        logits = self.classifier(x)          # shape: (batch, 1)
        probs = torch.sigmoid(logits.squeeze(-1) + self.shift)
        return torch.cat((probs.unsqueeze(-1), 1 - probs.unsqueeze(-1)), dim=-1)


__all__ = ["HybridNet", "ResidualBlock"]
