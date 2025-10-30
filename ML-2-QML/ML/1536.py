"""Enhanced quanvolution model with patch‑wise attention and data augmentation.

The classical version replaces the fixed 2×2 convolution with a learnable
attention scaling over the four output channels and augments each batch with
random flips and 90° rotations.  The forward pass returns logits suitable
for multi‑class classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple


class PatchAttentionConv(nn.Module):
    """
    Convolutional layer followed by a learnable per‑channel attention
    scaling.  The convolution reduces the 28×28 image to 14×14 patches,
    each represented by four channels.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Learnable attention weights for the four output channels
        self.attn = nn.Parameter(torch.randn(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        out = self.conv(x)  # (B, out_channels, 14, 14)
        # Scale each channel by a learned weight
        scale = torch.sigmoid(self.attn).view(1, -1, 1, 1)
        return out * scale


class DataAugmentation:
    """Random horizontal flip and rotation by multiples of 90° applied per batch."""
    @staticmethod
    def augment(x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-1])  # horizontal flip
        k = random.randint(0, 3)
        if k:
            x = torch.rot90(x, k, dims=[-2, -1])
        return x


class QuanvolutionFilter(nn.Module):
    """Classical quanvolution filter with attention and augmentation."""
    def __init__(self) -> None:
        super().__init__()
        self.attn_conv = PatchAttentionConv()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = DataAugmentation.augment(x)
        features = self.attn_conv(x)  # (B, 4, 14, 14)
        return features.view(x.size(0), -1)  # flatten to (B, 4*14*14)


class QuanvolutionClassifier(nn.Module):
    """Classifier head that follows the quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
