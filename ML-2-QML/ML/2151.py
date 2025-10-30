"""Enhanced classical quanvolution with depth‑wise separable structure and residual shortcut."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseQuanvolutionFilter(nn.Module):
    """Depth‑wise separable quanvolution.

    The filter operates on non‑overlapping 2×2 patches, applies a 2×2 depth‑wise kernel
    (one kernel per input channel), then mixes the resulting feature maps with a
    1×1 point‑wise convolution.  Dropout and a residual connection to the input
    are added for better regularisation and gradient flow.
    """
    def __init__(self, in_channels: int = 1, depthwise_kernel: tuple[int, int] = (2, 2),
                 dropout: float = 0.1, residual: bool = True) -> None:
        super().__init__()
        self.residual = residual
        self.dropout = nn.Dropout2d(dropout)

        # Depth‑wise convolution: one kernel per input channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=depthwise_kernel,
                                   stride=depthwise_kernel, groups=in_channels, bias=True)

        # Point‑wise (1×1) convolution to mix the channels
        self.pointwise = nn.Conv2d(in_channels, in_channels * 4, kernel_size=1,
                                   stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve original shape for residual
        residual = x if self.residual else 0

        # Depth‑wise convolution
        dw = self.depthwise(x)
        # Dropout on the depth‑wise output
        dw = self.dropout(dw)

        # Point‑wise mixing
        pw = self.pointwise(dw)

        # Upsample residual to match output shape
        if self.residual:
            # Input size is (N, C, 28, 28). After depth‑wise stride 2: (N, C, 14, 14)
            # After point‑wise: (N, 4C, 14, 14)
            residual = F.interpolate(residual, size=pw.shape[2:], mode='nearest')
            return pw + residual
        return pw


class QuanvolutionPlusClassifier(nn.Module):
    """Hybrid classifier using the enhanced depth‑wise quanvolution."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = DepthwiseQuanvolutionFilter()
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.classifier(features.view(x.size(0), -1))
        return F.log_softmax(logits, dim=-1)


__all__ = ["DepthwiseQuanvolutionFilter", "QuanvolutionPlusClassifier"]
