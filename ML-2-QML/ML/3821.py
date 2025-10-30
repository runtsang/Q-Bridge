"""Classical implementation of a hybrid quanvolution classifier.

The model uses depthwise‑separable convolutions followed by a
batch‑norm, dropout, and a linear head.  A lightweight sampler
network (MLP) is incorporated as a head in the quantum version.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybridClassifier(nn.Module):
    """
    Classical depthwise‑separable quanvolution classifier.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels (e.g., MNIST images).
    num_classes : int, default 10
        Number of target classes.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        # Depthwise convolution: one filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels
        )
        # Pointwise convolution to mix channels
        self.pointwise = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.bn = nn.BatchNorm2d(4)
        self.dropout = nn.Dropout2d(0.1)

        # Flattened feature size: 4 channels × 14 × 14 pixels
        self.fc = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybridClassifier"]
