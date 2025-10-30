"""ConvGen101: depth‑wise separable convolution with residual.

The class inherits the original Conv interface so that it can be dropped in
wherever Conv was used.  The new implementation adds a learnable depth‑wise
kernel followed by a point‑wise 1×1 convolution.  A residual shortcut is
included to preserve the original behaviour when the depth‑wise filter
has no effect.  The implementation is fully compatible with PyTorch’s
autograd and can be serialised by torch.save().
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple, Union

class ConvGen101(nn.Module):
    """
    Depth‑wise separable convolution with residual connection.

    Parameters
    ----------
    kernel_size : int or tuple
        Size of the depth‑wise and point‑wise kernels.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool, optional
        Whether to include a bias term in the point‑wise convolution.
    """

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 in_channels: int,
                 out_channels: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # depth‑wise convolution: one filter per input channel
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size=kernel_size,
                                   groups=in_channels,
                                   bias=False,
                                   padding=kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size))

        # point‑wise convolution: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   bias=bias)

        # residual shortcut (identity) if channel dimensions match
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=1,
                                      bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the depth‑wise separable convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C_in, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C_out, H, W).
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        # add residual
        out += self.residual(x)
        return out

    def get_params(self) -> dict:
        """Return a dictionary of all learnable parameters."""
        return {
            'depthwise': self.depthwise.weight,
            'pointwise': self.pointwise.weight,
            'bias': self.pointwise.bias if self.pointwise.bias is not None else None,
           'residual': getattr(self.residual, 'weight', None)
        }

__all__ = ["ConvGen101"]
