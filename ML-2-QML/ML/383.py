"""Enhanced classical convolutional classifier with residual and depthwise separable layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionPlus(nn.Module):
    """Depthwise separable conv + residual + linear head."""

    def __init__(self) -> None:
        super().__init__()
        # depthwise conv
        self.depthwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            bias=False,
        )
        # pointwise conv
        self.pointwise = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(4)
        # residual path
        self.residual_conv = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=1,
            stride=2,
            bias=False,
        )
        self.residual_bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        # linear head
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        res = self.residual_conv(x)
        res = self.residual_bn(res)
        out = out + res
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return F.log_softmax(logits, dim=-1)
