"""ConvEnhanced: multi‑channel, adaptive convolutional filter for classical deep learning."""

from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

def ConvEnhanced(kernel_size: int = 3,
                 channels: int = 1,
                 bias: bool = True,
                 threshold: float = 0.0,
                 use_batchnorm: bool = False,
                 dropout: float | None = None,
                 activation: str = "relu") -> nn.Module:
    """
    Return a torch.nn.Module that can be dropped into a standard CNN.

    Parameters
    ----------
    kernel_size : int, default 3
        Size of the convolution kernel.
    channels : int, default 1
        Number of input/output channels.
    bias : bool, default True
        Whether the convolution layer uses a bias term.
    threshold : float, default 0.0
        Threshold applied to the convolution output before activation.
    use_batchnorm : bool, default False
        Whether a BatchNorm2d layer is inserted after the convolution.
    dropout : float or None, default None
        Dropout probability applied after activation.
    activation : str, default "relu"
        Activation function name from torch.nn.functional.

    Returns
    -------
    nn.Module
        A convolutional filter that outputs a single scalar: the mean
        activation over the feature map.  The module can be used as a
        drop‑in replacement for the original Conv class while providing
        richer behaviour for deeper networks.
    """
    class ConvFilter(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels,
                                  kernel_size=kernel_size,
                                  bias=bias)
            self.threshold = threshold
            self.use_batchnorm = use_batchnorm
            self.batchnorm = nn.BatchNorm2d(channels) if use_batchnorm else None
            self.dropout = nn.Dropout(dropout) if dropout else None
            self.activation = getattr(F, activation)

        def forward(self, x: Tensor) -> float:
            out = self.conv(x)
            out = out - self.threshold
            out = self.activation(out)
            if self.batchnorm:
                out = self.batchnorm(out)
            if self.dropout:
                out = self.dropout(out)
            return out.mean().item()

    return ConvFilter()

__all__ = ["ConvEnhanced"]
