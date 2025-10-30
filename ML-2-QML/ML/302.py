"""Enhanced classical quanvolution with optional pooling and bias.

This module retains the original 2×2 patch extraction but adds
configurable pooling (max/avg) and an optional bias term for the
fully‑connected head.  The filter is fully compatible with
PyTorch autograd and can be used directly in a standard training
loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class QuanvolutionFilter(nn.Module):
    """Classical convolution‑like filter with optional pooling.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel.
    stride : int, default 2
        Stride of the convolution.
    pooling : str or None, default'max'
        Pooling operation applied after convolution.  Options are
        ``'max'``, ``'avg'`` or ``None`` for no pooling.
    bias : bool, default True
        Whether to include a bias term in the convolution.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 2,
                 pooling: Optional[str] ='max',
                 bias: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size,
                              stride=stride, bias=bias)
        if pooling not in (None,'max', 'avg'):
            raise ValueError("pooling must be None,'max' or 'avg'")
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply convolution, optional pooling, and flatten."""
        features = self.conv(x)
        if self.pooling =='max':
            features = F.max_pool2d(features, kernel_size=2)
        elif self.pooling == 'avg':
            features = F.avg_pool2d(features, kernel_size=2)
        # flatten to (batch, -1)
        return features.view(features.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quanvolutional filter followed by a linear head."""
    def __init__(self,
                 pooling: Optional[str] ='max',
                 bias: bool = True) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(pooling=pooling, bias=bias)
        # 4 output channels, after pooling the feature map is 14x14
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
