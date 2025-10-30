"""ConvEnhanced: 3‑D convolutional filter with optional residual connection.

This module extends the original Conv filter to support multi‑channel 3‑D data,
a residual skip connection, and a customizable activation.  It remains
drop‑in compatible: the ``Conv`` factory still returns a ``ConvEnhanced``
instance, so existing pipelines can use it unchanged.

The class is lightweight and depends only on PyTorch and NumPy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable

class ConvEnhanced(nn.Module):
    """3‑D convolutional filter with optional residual connection.

    Parameters
    ----------
    kernel_size : int
        Size of the cubic kernel.  For the original 2‑D case ``kernel_size=2``
        and ``channels=1``.
    channels : int, default 1
        Number of input and output channels.  When ``channels==1`` the module
        behaves like the original 2‑D filter.
    threshold : float, default 0.0
        Value subtracted from the convolution output before activation.
    activation : Callable[[torch.Tensor], torch.Tensor], default nn.Sigmoid()
        Activation applied after the threshold.  Pass ``None`` for no
        activation.
    skip : bool, default False
        If ``True`` the input is added to the convolution output before
        activation, mimicking a residual block.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        channels: int = 1,
        threshold: float = 0.0,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.Sigmoid(),
        skip: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.threshold = threshold
        self.activation = activation
        self.skip = skip

        if channels == 1 and kernel_size == 2:
            # Keep the exact behaviour of the original 2‑D filter
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
            self.is_3d = False
        else:
            # Use a 3‑D convolution for volumetric data
            self.conv = nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                bias=True,
            )
            self.is_3d = True

    def forward(self, data: np.ndarray) -> torch.Tensor:
        """Run the convolution on ``data``.

        Parameters
        ----------
        data : np.ndarray
            If ``channels==1`` the shape must be
            ``(kernel_size, kernel_size)``.  For multi‑channel data the shape
            should be ``(channels, depth, height, width)`` where
            ``depth == height == width == kernel_size``.

        Returns
        -------
        torch.Tensor
            The mean activation value across the output tensor.
        """
        if self.is_3d:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            if tensor.ndim!= 4:
                raise ValueError(
                    f"Expected 4‑D array for 3‑D convolution, got {tensor.ndim}‑D."
                )
            tensor = tensor.unsqueeze(0)  # add batch dimension
            conv_out = self.conv(tensor)
        else:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            if tensor.ndim!= 2:
                raise ValueError(
                    f"Expected 2‑D array for 2‑D convolution, got {tensor.ndim}‑D."
                )
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            conv_out = self.conv(tensor)

        if self.skip:
            conv_out = conv_out + tensor

        conv_out = conv_out - self.threshold

        if self.activation is not None:
            conv_out = self.activation(conv_out)

        return conv_out.mean()

    def run(self, data: np.ndarray) -> float:
        """Convenience wrapper that returns a Python float."""
        return float(self.forward(data).item())


def Conv() -> ConvEnhanced:
    """Factory for backward compatibility."""
    return ConvEnhanced()


__all__ = ["ConvEnhanced", "Conv"]
