"""Enhanced classical convolution filter with depth‑wise separable and multi‑channel support.

The Conv function returns a torch.nn.Module that mimics the behavior of the
quantum filter, but adds depth‑wise separable convolution, multi‑channel
input/output, and an optional thresholded sigmoid activation.  It is
fully differentiable and can be dropped into a larger PyTorch model.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

__all__ = ["Conv"]


def Conv(kernel_size: int = 3,
         in_channels: int = 1,
         out_channels: int = 1,
         depthwise: bool = True,
         threshold: float = 0.0) -> nn.Module:
    """Create a convolution filter inspired by the quantum quanvolution.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel (default 3).
    in_channels : int
        Number of input feature maps.
    out_channels : int
        Number of output feature maps.
    depthwise : bool
        If True use depth‑wise separable convolution (groups=in_channels).
        Otherwise use a standard convolution.
    threshold : float
        If >0.0 apply a sigmoid activation after subtracting ``threshold``.
        The output is then the mean over spatial dimensions.

    Returns
    -------
    nn.Module
        A PyTorch module with a single ``forward`` method.
    """
    class ConvFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if depthwise:
                # Depth‑wise convolution
                self.depthwise = nn.Conv2d(in_channels,
                                           in_channels,
                                           kernel_size=kernel_size,
                                           groups=in_channels,
                                           bias=True)
                # Point‑wise convolution to mix channels
                self.pointwise = nn.Conv2d(in_channels,
                                           out_channels,
                                           kernel_size=1,
                                           bias=True)
            else:
                self.depthwise = nn.Conv2d(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           bias=True)
                self.pointwise = None
            self.threshold = threshold

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : torch.Tensor
                Shape ``(N, C_in, H, W)``.

            Returns
            -------
            torch.Tensor
                Shape ``(N, C_out, H_out, W_out)``.  If ``threshold`` is set,
                a sigmoid activation is applied after subtracting the threshold
                and the output is the mean over ``H_out`` and ``W_out``.
            """
            if self.pointwise is None:
                y = self.depthwise(x)
            else:
                y = self.depthwise(x)
                y = self.pointwise(y)
            if self.threshold!= 0.0:
                y = torch.sigmoid(y - self.threshold)
                # Return mean per sample
                return y.mean(dim=[2, 3])
            return y

    return ConvFilter()
