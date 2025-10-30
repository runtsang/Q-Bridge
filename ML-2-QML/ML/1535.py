"""Hybrid classical neural network for image classification.

The module contains a depth‑wise separable convolution that captures local
2×2 patterns, a residual block to learn higher‑level features, and a
multi‑layer perceptron head for classification.  It can be trained on CPU
or GPU with standard PyTorch pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """
    Classical hybrid network inspired by the original quanvolution filter.
    Consists of a depth‑wise separable convolution that processes 2×2 patches,
    followed by a residual stack and a multi‑layer perceptron head.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 32) -> None:
        super().__init__()
        # Depth‑wise conv to capture local 2×2 patterns
        self.depthwise = nn.Conv2d(
            in_channels, base_channels, kernel_size=2, stride=2, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Residual block
        self.residual = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        # Flatten and MLP head
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(base_channels * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.bn1(out)
        res = self.residual(out)
        out = self.relu(out + res)
        out = self.flatten(out)
        logits = self.mlp(out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
