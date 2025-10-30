"""Enhanced classical CNN with depth‑wise separable convs and residuals.

QFCModel is a drop‑in replacement for the original seed model. It now
supports richer feature extraction via depth‑wise separable convolutions
and a lightweight residual connection. The final head is fully
connected with dropout and group norm, enabling better regularisation
on small batches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depth‑wise separable convolution block.

    This consists of a depth‑wise convolution followed by a point‑wise
    1×1 convolution. It reduces the number of parameters while
    preserving representational power.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.GroupNorm(num_groups=8, num_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Simple residual block using a depth‑wise separable conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = DepthwiseSeparableConv(channels, channels)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x)
        out = self.dropout(out)
        return F.relu(out + residual)


class QFCModel(nn.Module):
    """Classical CNN → FC model with separable convs + residuals."""

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            DepthwiseSeparableConv(1, 16, kernel_size=3, padding=1),
            ResidualBlock(16),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(16, 32, kernel_size=3, padding=1),
            ResidualBlock(32),
            nn.MaxPool2d(2),
        )
        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 4),
        )
        self.norm = nn.GroupNorm(num_groups=4, num_channels=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28).

        Returns:
            Normalised logits of shape (batch, 4).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.norm(x)


__all__ = ["QFCModel"]
