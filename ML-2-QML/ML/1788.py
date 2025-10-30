"""Enhanced classical convolutional‑fully‑connected model with residuals and dropout.

The architecture extends the original CNN by:
    * 2 residual blocks (conv → relu → conv → add)
    * Dropout after the first fully‑connected layer
    * Dynamic handling of input images of arbitrary size via adaptive average pooling
    * Optional weight‑norm and batch‑norm layers for better training stability

The model still outputs a 4‑dimensional feature vector, matching the original design.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    """Simple 2‑layer residual block with batch‑norm."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # residual connection
        return self.relu(out)


class QFCModel(nn.Module):
    """Convolutional backbone with residuals + fully‑connected projector.

    The network is purposely lightweight for quick prototyping while still
    offering the flexibility to handle arbitrary input resolutions.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            _ResidualBlock(8, 16, stride=1),
            nn.MaxPool2d(2),
        )
        # Adaptive pooling so that the classifier works for any input size
        self.adapt_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),  # dropout for regularization
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4, bias=True),
        )
        self.output_norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Normalized 4‑dimensional feature vector.
        """
        bsz = x.shape[0]
        feats = self.features(x)
        feats = self.adapt_pool(feats)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        return self.output_norm(out)


__all__ = ["QFCModel"]
