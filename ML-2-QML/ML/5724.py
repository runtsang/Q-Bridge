"""Hybrid classical‑quantum style model with an extended, trainable filter and skip‑connections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]

class QuanvolutionFilter(nn.Module):
    """
    A depth‑wise separable convolution with a residual connection.
    Each input channel is convolved separately (groups=in_channels) and then
    a 1×1 point‑wise convolution mixes the channels. The residual block
    adds the original input to the output (after a 1×1 projection if the
    number of channels changes) to preserve low‑frequency information.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2,
                 stride: int = 2, padding: int = 0) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=stride, bias=False) if in_channels!= out_channels else None
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        dw = self.depthwise(x)
        pw = self.pointwise(dw)
        residual = self.res_proj(x) if self.res_proj is not None else x
        out = pw + residual
        out = self.bn(out)
        return out.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """
    Hybrid network that uses the extended QuanvolutionFilter followed by a
    fully‑connected head. The filter produces a feature vector that is
    fed into a linear classifier.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1,
                 filter_channels: int = 4) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels=in_channels,
                                          out_channels=filter_channels)
        self.linear = nn.Linear(filter_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)
