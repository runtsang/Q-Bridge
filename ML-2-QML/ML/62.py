"""Enhanced classical convolutional filter with depth‑wise separable support.

This module defines :class:`ConvEnhanced`, a drop‑in replacement for the
original ``Conv()`` factory.  It supports arbitrary input/output channels,
depth‑wise separable convolution, a learnable threshold, and batched data
processing.  The class exposes a ``forward`` method for training and a
``run`` convenience wrapper that accepts NumPy arrays or PyTorch tensors.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ConvEnhanced(nn.Module):
    """
    Depth‑wise separable convolutional filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel.
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels.
    stride : int, default 1
        Stride of the convolution.
    separable : bool, default True
        If True, use depth‑wise followed by point‑wise convolution.
    learnable_threshold : bool, default True
        If True, the threshold is a learnable parameter.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        stride: int = 1,
        separable: bool = True,
        learnable_threshold: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.separable = separable

        if separable:
            # Depth‑wise convolution
            self.depthwise = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=in_channels,
                bias=False,
            )
            # Point‑wise convolution
            self.pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=True,
            )

        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("threshold", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Mean activation per sample after sigmoid and threshold,
            shape (batch,).
        """
        if self.separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)

        logits = x
        activations = torch.sigmoid(logits - self.threshold)
        # Mean over spatial and channel dimensions
        return activations.mean(dim=[1, 2, 3])

    def run(self, data):
        """
        Convenience wrapper that accepts a NumPy array or torch tensor.

        Parameters
        ----------
        data : array-like
            Input data of shape (batch, channels, height, width) or
            (height, width) for a single sample.

        Returns
        -------
        float
            Mean activation over the batch.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)

        # Normalize dimensions
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif data.ndim == 3 and data.shape[0] == 1:
            data = data.unsqueeze(1)  # (1, C, H, W)

        return self.forward(data).item()


__all__ = ["ConvEnhanced"]
