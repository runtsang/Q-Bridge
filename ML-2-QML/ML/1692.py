"""
Classical hybrid filter inspired by the original Quanvolution example.
This version uses a depthwise separable 2×2 convolution followed by a
point‑wise 1×1 convolution, batch‑norm, ReLU and dropout.  The output
features are flattened and fed into a linear classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQuanvolutionFilter(nn.Module):
    """
    Depthwise‑separable 2×2 convolution that produces 4 feature maps per
    patch.  The 2×2 kernel acts like a classical “quanvolution” patch
    extractor but with trainable weights, batch‑norm, ReLU, and dropout.
    """
    def __init__(self) -> None:
        super().__init__()
        # Depth‑wise 2×2 conv (stride 2 reduces 28×28 → 14×14)
        self.depthwise = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=2,
            stride=2,
            groups=1,
            bias=False,
        )
        # Point‑wise 1×1 conv to mix the 4 feature maps
        self.pointwise = nn.Conv2d(
            in_channels=4,
            out_channels=4,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, 28, 28)

        Returns:
            Tensor of shape (B, 4*14*14) ready for the linear head.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x.view(x.size(0), -1)

class ClassicalQuanvolutionClassifier(nn.Module):
    """
    Purely classical classifier that mirrors the original architecture.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["ClassicalQuanvolutionFilter", "ClassicalQuanvolutionClassifier"]
