"""Enhanced classical quanvolutional model with depthwise‑separable convolution and batch‑norm."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionModel(nn.Module):
    """
    Classical convolutional model inspired by the original quanvolution example.
    Adds depthwise‑separable convolutions, batch normalization, and a
    configurable linear classifier head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        depth_multiplier: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size=2,
            stride=2,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels * depth_multiplier,
            in_channels * depth_multiplier,
            kernel_size=1,
        )
        self.bn = nn.BatchNorm2d(in_channels * depth_multiplier)
        # Feature dimension after 2x2 stride convolution on a 28x28 image
        self.feature_dim = (28 // 2) * (28 // 2) * in_channels * depth_multiplier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, H, W)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = F.relu(self.bn(x))
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionModel"]
