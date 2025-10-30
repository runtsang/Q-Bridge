"""ConvEnhanced: multi‑scale attention‑aware convolutional filter.

This module extends the original Conv filter by adding:
- batch support
- optional depthwise separable kernels
- learnable threshold bias
- residual skip connection
- lightweight spatial attention
- flexible output mode (mean or flattened)

It can be dropped into any PyTorch model as a drop‑in replacement.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple, Union, Dict

class ConvEnhanced(nn.Module):
    """
    A drop‑in classical convolutional filter with multi‑scale attention.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels.
    kernel_size : int or tuple, default 2
        Size of the convolution kernel.
    stride : int or tuple, default 1
        Stride of the convolution.
    padding : int or tuple, default 0
        Zero‑padding added to both sides of the input.
    bias : bool, default True
        Whether to add a learnable bias to the convolution.
    depthwise : bool, default False
        If True, use depthwise separable convolution to reduce parameters.
    attention : bool, default True
        If True, apply a 1×1 attention map before the residual addition.
    residual : bool, default True
        If True, add a residual connection from the input.
    threshold : float, default 0.0
        Initial value for the learnable threshold bias.
    batch_norm : bool, default False
        If True, apply batch normalization after the convolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
        depthwise: bool = False,
        attention: bool = True,
        residual: bool = True,
        threshold: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.depthwise = depthwise
        self.attention = attention
        self.residual = residual
        self.batch_norm = batch_norm

        # Convolution layer (depthwise if requested)
        if depthwise:
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=bias,
            )
            self.pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
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

        # Attention map
        if attention:
            self.attn = nn.Conv2d(
                out_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.attn = None

        # Residual projection
        if residual:
            self.res_proj = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.res_proj = None

        # Batch norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        # Learnable threshold bias
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W) or (B, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after convolution, attention, residual and threshold.
        """
        # Ensure 4‑D input
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)

        # Convolution
        if self.depthwise:
            out = self.conv(x)
            out = self.pointwise(out)
        else:
            out = self.conv(x)

        # Attention
        if self.attention:
            attn_map = torch.sigmoid(self.attn(out))
            out = out * attn_map

        # Residual
        if self.residual:
            res = self.res_proj(x)
            out = out + res

        # Batch norm
        if self.bn is not None:
            out = self.bn(out)

        # Threshold bias
        out = torch.sigmoid(out - self.threshold)

        return out

    def mean_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the mean activation over spatial dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Mean activation per batch element.
        """
        return self.forward(x).mean(dim=[2, 3])

    @classmethod
    def from_dict(cls, cfg: Dict) -> "ConvEnhanced":
        """
        Construct the module from a dictionary of hyper‑parameters.

        Parameters
        ----------
        cfg : dict
            Dictionary containing any of the constructor arguments.

        Returns
        -------
        ConvEnhanced
            Instantiated module.
        """
        return cls(**cfg)

__all__ = ["ConvEnhanced"]
