"""Hybrid NAT model – classical implementation.

The model consists of a CNN backbone, a residual feature extractor
that mimics the non‑linear power of a variational quantum circuit,
and a linear head.  It is compatible with both classification and
regression tasks via the ``output_dim`` parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridNATModel"]


class HybridNATModel(nn.Module):
    """CNN backbone + residual layer + linear head.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels (e.g. 1 for grayscale images).
    output_dim : int, default 4
        Dimensionality of the output.  Use ``1`` for regression.
    """

    def __init__(self, in_channels: int = 1, output_dim: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Residual block that simulates a quantum circuit
        self.residual = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(64, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x).view(x.size(0), -1)
        features = self.residual(features)
        out = self.head(features)
        return self.bn(out)
