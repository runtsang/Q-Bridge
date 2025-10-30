"""Hybrid classical convolutional filter with depth‑wise separable conv and residual scaling.

This module provides a drop‑in replacement for the original Conv filter but
adds a depth‑wise separable convolution, a learnable residual scaling
factor, and optional L2 weight decay.  The public API is the class
``ConvFilter`` and the helper function ``Conv`` that returns an instance.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """
    Hybrid classical convolutional filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    threshold : float, default 0.0
        Bias threshold used in the original implementation.
    separable : bool, default True
        If True, use depth‑wise separable convolution.
    residual : bool, default True
        If True, add a learnable residual scaling factor.
    weight_decay : float | None, default None
        If provided, apply L2 weight decay to the convolution weights.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        separable: bool = True,
        residual: bool = True,
        weight_decay: float | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.separable = separable
        self.residual = residual

        if self.separable:
            # depth‑wise convolution (1 input channel → 1 output channel)
            self.depth = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=False,
            )
            # point‑wise convolution (1 channel → 1 channel)
            self.point = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True,
            )

        if self.residual:
            # learnable scaling of the input residual
            self.scale = nn.Parameter(torch.ones(1))

        if weight_decay is not None:
            # register weight decay as a parameter so that optimizers
            # can apply it automatically.
            self.weight_decay = weight_decay
        else:
            self.weight_decay = None

        # initialise bias to the threshold value
        if self.separable:
            self.point.bias.data.fill_(self.threshold)
        else:
            self.conv.bias.data.fill_(self.threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the filter.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (1, 1, kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Scalar output after sigmoid activation.
        """
        if self.separable:
            out = self.depth(x)
            out = self.point(out)
        else:
            out = self.conv(x)

        if self.residual:
            out = out + self.scale * x

        # apply sigmoid and return mean activation
        return torch.sigmoid(out - self.threshold).mean()

    def run(self, data) -> float:
        """
        Convenience method compatible with the original API.

        Parameters
        ----------
        data : array‑like or torch.Tensor
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Scalar output of the filter.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        with torch.no_grad():
            return self.forward(tensor).item()

    def export_kernel(self) -> torch.Tensor:
        """
        Return the effective kernel learned by the filter.

        Returns
        -------
        torch.Tensor
            Kernel of shape (1, 1, kernel_size, kernel_size).
        """
        if self.separable:
            depth_w = self.depth.weight
            point_w = self.point.weight
            # point_w is 1x1, so broadcast
            kernel = depth_w * point_w
        else:
            kernel = self.conv.weight
        return kernel.clone().detach()

def Conv(*args, **kwargs):
    """
    Backward‑compatible factory that returns a ``ConvFilter`` instance.
    """
    return ConvFilter(*args, **kwargs)

__all__ = ["ConvFilter", "Conv"]
