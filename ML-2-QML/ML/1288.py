"""Enhanced classical quanvolution model with residual connection and batch normalization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionNet(nn.Module):
    """Learnable classical quanvolution network."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Residual connection from input to output of conv
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.classifier = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        res_out = self.residual(x)
        out = self.relu(conv_out + res_out)
        out = self.bn(out)
        features = out.view(out.size(0), -1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionNet"]
