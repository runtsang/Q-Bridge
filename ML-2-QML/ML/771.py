"""Enhanced classical convolution filter with batch‑norm, dropout, and global pooling.

The class is a drop‑in replacement for the original Conv filter.  It
accepts a 2‑D kernel of arbitrary size and returns a scalar
activation.  Internally it uses a learnable 2‑D convolution followed
by batch‑normalisation, a sigmoid threshold, optional dropout, and
global‑average pooling.  The ``run`` method keeps the original
signature for backward compatibility.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import pad


class ConvFilter(nn.Module):
    """
    A richer classical convolution filter.

    Parameters
    ----------
    kernel_size : int, default 3
        Size of the square kernel.
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 16
        Number of output channels after convolution.
    stride : int, default 1
        Convolution stride.
    padding : int, default 1
        Zero‑padding added to both sides of the input.
    threshold : float, default 0.0
        Value subtracted before the sigmoid activation.
    dropout : float, default 0.0
        Dropout probability applied after activation.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        in_channels: int = 1,
        out_channels: int = 16,
        stride: int = 1,
        padding: int = 1,
        threshold: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = torch.sigmoid(x - self.threshold)
        x = self.dropout(x)
        # Global average pooling to a scalar
        return x.mean(dim=[2, 3])

    def run(self, data) -> float:
        """
        Run the filter on a 2‑D kernel and return a scalar.

        Parameters
        ----------
        data : array‑like
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Scalar activation value.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        # Ensure the tensor has the expected shape; pad if necessary
        if tensor.shape!= (self.kernel_size, self.kernel_size):
            # Pad to the required size
            pad_left = (self.kernel_size - tensor.shape[1]) // 2
            pad_top = (self.kernel_size - tensor.shape[0]) // 2
            tensor = pad(tensor, (pad_left, self.kernel_size - tensor.shape[1] - pad_left,
                                 pad_top, self.kernel_size - tensor.shape[0] - pad_top))
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (B, C, H, W)
        out = self.forward(tensor)
        return out.item()


def Conv(kernel_size: int = 3,
         in_channels: int = 1,
         out_channels: int = 16,
         stride: int = 1,
         padding: int = 1,
         threshold: float = 0.0,
         dropout: float = 0.0) -> ConvFilter:
    """
    Factory that returns a configured ConvFilter instance.
    """
    return ConvFilter(
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        padding=padding,
        threshold=threshold,
        dropout=dropout,
    )


__all__ = ["Conv"]
