"""Enhanced classical convolution module with multi‑channel support and optional pooling.

Provides a ConvFilter class that mimics the quantum filter interface and can be used as a drop‑in replacement for the original `Conv` function."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvFilter(nn.Module):
    """
    A classical 2‑D convolution filter that mirrors the behaviour of the original
    quantum filter.  The module accepts multi‑channel data, supports configurable
    stride/padding, and can apply optional max‑pooling after the convolution.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.0,
        pool: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square convolution kernel.
        in_channels : int
            Number of input feature maps.
        out_channels : int
            Number of output feature maps produced by the filter.
        stride : int
            Stride of the convolution.
        padding : int
            Padding added to all sides.
        threshold : float
            Activation threshold used in the sigmoid.
        pool : str | None
            Optional max‑pooling after the convolution; pass '2x2' or None.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.pool = pool

        # Learnable weight tensor; bias is optional.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the convolution followed by a sigmoid threshold."""
        out = self.conv(x)
        activations = F.sigmoid(out - self.threshold)
        if self.pool == "2x2":
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            out = pool(activations)
            return out
        return activations

    def run(self, data: torch.Tensor) -> float:
        """
        Convenience wrapper that accepts a raw NumPy array or a torch tensor,
        feeds it through the filter, and returns the mean activation value.
        """
        if isinstance(data, torch.Tensor):
            tensor = data
        else:
            tensor = torch.as_tensor(data, dtype=torch.float32)
        out = self.forward(tensor)
        return out.mean().item()
