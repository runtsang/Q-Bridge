"""ConvEnhanced: Classical convolutional filter with advanced features.

Features:
- Multi‑channel depth‑wise separable convolution.
- Learnable threshold applied after convolution.
- Optional residual connection.
- Drop‑in replacement for the original Conv filter.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

class ConvEnhanced(nn.Module):
    """
    A versatile convolutional filter that extends the original 2‑D filter.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels.
    kernel_size : int or Tuple[int, int], default 2
        Size of the convolution kernel.
    stride : int, default 1
        Stride of the convolution.
    padding : int, default 0
        Zero‑padding added to both sides of the input.
    bias : bool, default True
        If ``True``, adds a learnable bias to the convolution.
    separable : bool, default False
        If ``True`` a depth‑wise separable convolution is used.
    residual : bool, default False
        If ``True`` a residual connection is added when shapes match.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int | Tuple[int, int] = 2,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        separable: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.separable = separable
        self.residual = residual

        if separable:
            # depth‑wise convolution (groups == in_channels)
            self.depthwise = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=in_channels,
            )
            # point‑wise convolution
            self.pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )

        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Output tensor after convolution, thresholding and optional residual.
        """
        x_in = x
        if self.separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)

        # Apply thresholded sigmoid activation
        x = torch.sigmoid(x - self.threshold)

        # Optional residual connection
        if self.residual and x.shape == x_in.shape:
            x = x + x_in

        return x

    def run(self, data):
        """
        Convenience wrapper that accepts a NumPy array or a list of lists
        and returns the mean activation value.

        Parameters
        ----------
        data : array‑like
            Input data of shape ``(H, W)`` for a single sample.

        Returns
        -------
        float
            Mean of the thresholded activations.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        # Ensure shape (1, in_channels, H, W)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] == self.in_channels:
            tensor = tensor.unsqueeze(0)
        else:
            raise ValueError("Input shape is not compatible with the filter.")
        out = self.forward(tensor)
        return out.mean().item()

__all__ = ["ConvEnhanced"]
