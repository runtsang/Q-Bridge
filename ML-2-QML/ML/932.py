"""Hybrid classical convolution module with depth‑wise separable support and multi‑channel capability.

This module is fully differentiable in PyTorch and can be plugged into larger CNNs.  It also exposes a
`set_kernel` API that lets the user replace the learned weights with a pre‑trained kernel.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple


class ConvGen213(nn.Module):
    """
    Drop‑in replacement for the original Conv filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels.  If ``out_channels`` equals ``in_channels`` and
        ``depthwise`` is True, a depth‑wise separable convolution is used.
    depthwise : bool, default True
        Whether to use a depth‑wise separable convolution.
    threshold : float, default 0.0
        Threshold for the sigmoid activation.
    bias : bool, default True
        Whether the convolution includes a bias term.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        depthwise: bool = True,
        threshold: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.depthwise = depthwise

        if depthwise:
            # Depth‑wise separable convolution
            self.conv_dw = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                bias=bias,
            )
            if out_channels!= in_channels:
                self.conv_pw = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=bias,
                )
            else:
                self.conv_pw = None
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W) or (C, H, W).

        Returns
        -------
        torch.Tensor
            Output after convolution and sigmoid activation.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)  # add batch dimension

        if self.depthwise:
            out = self.conv_dw(x)
            if self.conv_pw is not None:
                out = self.conv_pw(out)
        else:
            out = self.conv(x)

        out = torch.sigmoid(out - self.threshold)
        return out

    def run(self, data: torch.Tensor | list | tuple) -> float:
        """
        Convenience wrapper that accepts raw data and returns the mean activation.

        Parameters
        ----------
        data : array‑like
            Input data of shape (C, H, W) or (H, W). If a list/tuple is provided,
            it is converted to a torch tensor.

        Returns
        -------
        float
            Mean activation value.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        out = self.forward(data)
        return out.mean().item()

    def set_kernel(self, kernel: torch.Tensor) -> None:
        """
        Replace the convolution kernel with a user‑supplied tensor.

        Parameters
        ----------
        kernel : torch.Tensor
            Tensor of shape (out_channels, in_channels, kernel_size, kernel_size).
        """
        if self.depthwise:
            self.conv_dw.weight.data = kernel[:, :, :, :]
        else:
            self.conv.weight.data = kernel[:, :, :, :]
