"""Enhanced classical quanvolution network with depthwise separable convolutions and residual connections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionEnhanced(nn.Module):
    """Depthwise separable convolution followed by a residual projection and a linear head."""

    def __init__(self) -> None:
        super().__init__()
        # Depthwise convolution: keeps channel count 1
        self.depthwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            stride=2,
            groups=1,
        )
        # Pointwise convolution: expands to 4 channels
        self.pointwise = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=1,
        )
        # Residual projection to match channel dimension
        self.res_proj = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=1,
        )
        # Linear head
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (batch, 1, 28, 28)
        dw = self.depthwise(x)  # (batch, 1, 14, 14)
        pw = self.pointwise(dw)  # (batch, 4, 14, 14)
        res = self.res_proj(x)  # (batch, 4, 14, 14)
        out = pw + res  # residual addition
        features = out.view(x.size(0), -1)  # flatten
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionEnhanced"]
