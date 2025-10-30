"""ConvGen155: enriched classical convolution module for hybrid workflows.

This module extends the original Conv filter by supporting multi‑channel,
depth‑wise separable convolutions, configurable activation functions, and a
utility to map the result to a probability distribution that can be fed
into a quantum backend for hybrid training.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Callable

__all__ = ["ConvGen155"]


class ConvGen155(nn.Module):
    """
    Drop‑in replacement for the original Conv filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter.
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 2
        Number of output channels (default 2 to allow feature splitting).
    depthwise : bool, default False
        If True, apply a depth‑wise separable convolution.
    bias : bool, default True
        Whether the convolution layers include a bias term.
    activation : Callable[[torch.Tensor], torch.Tensor], default nn.Sigmoid()
        Activation applied after the convolution.
    dropout : float, default 0.0
        Dropout probability applied after activation.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        depthwise: bool = False,
        bias: bool = True,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.Sigmoid(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if depthwise:
            # depth‑wise separable: one conv per input channel
            self.depthwise_conv = nn.Conv2d(
                in_channels, in_channels, kernel_size, bias=bias, groups=in_channels
            )
            self.pointwise_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=bias
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, bias=bias
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, H - k + 1, W - k + 1).
        """
        if self.depthwise:
            x = self.depthwise_conv(x)
            x = self.pointwise_conv(x)
        else:
            x = self.conv(x)

        x = self.activation(x)
        x = self.dropout(x)
        return x

    def to_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a real‑valued tensor to a probability distribution per spatial
        location by applying a sigmoid and normalising across channels.

        Parameters
        ----------
        x : torch.Tensor
            Tensor output of the convolution.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape with values in [0, 1] that sum to 1
            across the channel dimension at each spatial location.
        """
        # Sigmoid to [0,1]
        probs = torch.sigmoid(x)
        # Normalise across channels
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return probs
