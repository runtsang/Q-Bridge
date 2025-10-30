"""Enhanced classical convolutional filter with learnable threshold, batch normalization, and multi‑channel support.

The class can be used inside any PyTorch model. It exposes a `run` method that accepts a NumPy array or torch tensor and returns a probability‑like score.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ConvEnhanced(nn.Module):
    """
    Drop‑in replacement for the original Conv filter.

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
        If True, adds a learnable bias to the convolution.
    threshold_init : float, default 0.0
        Initial value for the learnable threshold.
    use_batchnorm : bool, default False
        If True, adds a BatchNorm2d layer after the convolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        threshold_init: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        # Learnable threshold that is optimised together with the rest of the network
        self.threshold = nn.Parameter(torch.tensor(threshold_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the filter.

        The convolution output is shifted by the learnable threshold before
        the sigmoid activation.  The result is a tensor with the same shape
        as the convolution output.
        """
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        out = out - self.threshold
        out = torch.sigmoid(out)
        return out

    def run(self, data) -> float:
        """
        Convenience method that accepts a NumPy array or torch tensor and
        returns a scalar probability‑like score.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Input data with shape (H, W) or (C, H, W).  If the input is
            two‑dimensional it is treated as a single‑channel image.

        Returns
        -------
        float
            Mean activation value over all spatial locations and output
            channels.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif data.ndim == 3:
            data = data.unsqueeze(0)  # (1, C, H, W)
        out = self.forward(data)
        return out.mean().item()


__all__ = ["ConvEnhanced"]
