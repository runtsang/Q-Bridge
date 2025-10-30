"""Conv: classical convolutional filter with optional depthwise separable support.

This module extends the original 1×1 filter to a full convolutional layer
that can handle multiple input and output channels.  It remains
drop‑in compatible: calling Conv() creates an instance with
a single output channel and a kernel size of 2, just like the seed.
The `run` method reproduces the original behaviour (mean sigmoid of the
logits minus a threshold) while the `forward` method exposes the
full feature map for use in larger CNNs.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Conv(nn.Module):
    """
    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels (feature maps).
    kernel_size : int or tuple, default 2
        Size of the convolution kernel.
    stride : int or tuple, default 1
        Stride of the convolution.
    padding : int or tuple, default 0
        Padding on each side.
    depthwise : bool, default False
        If True, use depthwise‑separable convolution.
    bias : bool, default True
        Whether to include a bias term.
    threshold : float, default 0.0
        Threshold applied to the logits before the sigmoid.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int | tuple = 2,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        depthwise: bool = False,
        bias: bool = True,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.depthwise = depthwise
        self.threshold = threshold

        # Build convolution weights
        if depthwise:
            self.weight = nn.Parameter(
                torch.empty(in_channels, 1, *self._size(kernel_size))
            )
            self.bias_param = nn.Parameter(
                torch.empty(in_channels)
            ) if bias else None
        else:
            self.weight = nn.Parameter(
                torch.empty(out_channels, in_channels, *self._size(kernel_size))
            )
            self.bias_param = nn.Parameter(
                torch.empty(out_channels)
            ) if bias else None

        # initialise
        nn.init.kaiming_uniform_(self.weight, a=0.2)
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)

    def _size(self, size):
        return size if isinstance(size, tuple) else (size, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: returns the full convolved feature map.
        """
        if self.depthwise:
            return F.conv2d(
                x,
                self.weight,
                bias=self.bias_param,
                stride=self.stride,
                padding=self.padding,
                groups=self.in_channels,
            )
        else:
            return F.conv2d(
                x,
                self.weight,
                bias=self.bias_param,
                stride=self.stride,
                padding=self.padding,
            )

    def run(self, data: torch.Tensor) -> float:
        """
        Compatibility method that mimics the original ConvFilter.
        Computes the mean sigmoid activation across all output
        elements after subtracting the threshold.
        """
        x = data.reshape(1, self.in_channels, *self._size(self.kernel_size))
        logits = self.forward(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


def ConvFactory() -> Conv:
    """Convenience factory that returns a Conv instance
    matching the original seed's behaviour: 1 channel, 1 output,
    kernel size 2, no threshold.
    """
    return Conv()


__all__ = ["Conv", "ConvFactory"]
