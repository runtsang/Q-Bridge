"""Enhanced classical convolutional filter with residual connections and channel attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quanvolution__gen124(nn.Module):
    """Classical quanvolution classifier with residual blocks and attention.

    The model processes 28×28 grayscale images. Two convolutional layers
    reduce the spatial dimensions to 7×7 while increasing feature depth.
    A residual connection allows gradients to flow more easily, and a
    channel‑attention module re‑weights the feature maps before the linear
    head.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        # Residual shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
        )

        # Channel‑attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=1),
            nn.Sigmoid(),
        )

        # Classifier head
        self.classifier = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Feature extraction with residual connection
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        skip = self.shortcut(x)
        out = out + skip

        # Attention re‑weighting
        attn = self.attention(out)
        out = out * attn

        # Flatten and classify
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution__gen124"]
