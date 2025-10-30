"""Enhanced Conv: a multi‑channel, gated, quantized convolution module for classical pipelines."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Conv(nn.Module):
    """
    Drop‑in replacement for the original Conv filter with richer functionality.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel.
    in_channels : int, default 1
        Number of input feature maps.
    out_channels : int, default 1
        Number of output feature maps.
    groups : int, default 1
        Number of convolution groups; when equal to in_channels, performs depth‑wise conv.
    bias : bool, default True
        Whether to add a bias term.
    gate : bool, default False
        If True, a residual sigmoid gate is applied to the conv output.
    quantize : bool, default False
        If True, the output is quantized to a binary representation (0/1) before returning.
    threshold : float, default 0.0
        Threshold used for the sigmoid activation and for the quantizer.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        groups: int = 1,
        bias: bool = True,
        gate: bool = False,
        quantize: bool = False,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.bias = bias
        self.gate = gate
        self.quantize = quantize
        self.threshold = threshold

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            groups=self.groups,
            bias=self.bias,
        )
        if self.gate:
            self.gate_layer = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional filter.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after optional gating and quantization.
        """
        out = self.conv(x)

        if self.gate:
            gated = self.gate_layer(out)
            out = out * gated

        if self.quantize:
            out = torch.where(out > self.threshold, torch.tensor(1.0, device=out.device), torch.tensor(0.0, device=out.device))

        return out

    def run(self, data):
        """
        Compatibility wrapper for the original API.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean activation value after optional gating and quantization.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        # Add batch and channel dimensions
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        out = self.forward(tensor)
        return out.mean().item()


__all__ = ["Conv"]
