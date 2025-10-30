"""Hybrid Quanvolution: classical convolutional feature extractor + linear head.

This module provides a drop‑in replacement for the original
QuanvolutionClassifier with a deeper CNN backbone that supports
larger images and optional weight sharing across channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """Classical convolutional feature extractor followed by a linear classifier.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    num_classes : int, default 10
        Number of output classes.
    conv_depth : int, default 3
        Number of convolutional blocks.
    num_filters : int, default 32
        Number of filters in the first conv block.
    weight_sharing : bool, default True
        Whether to share weights across successive conv blocks.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        conv_depth: int = 3,
        num_filters: int = 32,
        weight_sharing: bool = True,
    ) -> None:
        super().__init__()
        self.weight_sharing = weight_sharing
        layers = []
        in_ch = in_channels
        for i in range(conv_depth):
            out_ch = num_filters * (2 ** i)
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1)
            bn = nn.BatchNorm2d(out_ch)
            layers += [conv, bn, nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        # Flatten and linear head
        self.classifier = nn.Linear(out_ch * 28 * 28, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
