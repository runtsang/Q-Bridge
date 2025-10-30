"""Enhanced classical quanvolution network with trainable features and residual connections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class Quanvolution__gen329(nn.Module):
    """
    Classical quanvolution network that replaces the fixed convolution with a learnable
    filter and adds a residual connection from the input to the feature map.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        conv_out = self.conv(x)  # (batch, 4, 14, 14)
        # residual: upsample input to match shape
        residual = F.interpolate(x, size=conv_out.shape[2:], mode="nearest") * self.res_scale
        conv_out += residual
        features = conv_out.view(x.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["Quanvolution__gen329"]
