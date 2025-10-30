"""Hybrid classical convolutional block with residual and batch‑norm support.

The public API mimics the original ``Conv`` seed: a callable ``ConvGen158()`` that
returns an ``nn.Module``.  The returned module is fully‑trainable and can be
plugged into deeper CNNs.  A ``run`` method is kept for backward compatibility
and simply forwards the input through the convolution, optional batch‑norm,
activation and residual shortcut.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["ConvGen158"]


def ConvGen158(kernel_size: int = 2,
               bias: bool = True,
               use_bn: bool = False,
               residual: bool = False,
               threshold: float = 0.0) -> nn.Module:
    """
    Return a residual‑convolutional block with optional batch‑normalisation.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel.
    bias : bool, default True
        Whether to include a bias term in the convolution.
    use_bn : bool, default False
        If True a ``BatchNorm2d`` is applied after the convolution.
    residual : bool, default False
        If True a 1×1 shortcut projection is added.
    threshold : float, default 0.0
        Back‑compatibility parameter used only when ``use_bn`` is True.
        It is subtracted from the logits before the sigmoid activation.
    """

    class ConvFilter(nn.Module):
        """Learnable convolutional filter with optional batch‑norm and residual."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)
            self.use_bn = use_bn
            self.residual = residual
            if use_bn:
                self.bn = nn.BatchNorm2d(1)
            if residual:
                self.proj = nn.Conv2d(1, 1, kernel_size=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the block.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (N, 1, H, W).

            Returns
            -------
            torch.Tensor
                Output tensor of shape (N, 1, H-k+1, W-k+1).
            """
            out = self.conv(x)
            if self.use_bn:
                out = self.bn(out)
                out = out - threshold
            out = torch.sigmoid(out)
            if self.residual:
                shortcut = self.proj(x)
                out = out + shortcut
            return out

        def run(self, data: torch.Tensor) -> float:
            """
            Compatibility wrapper that returns the mean activation value.

            Parameters
            ----------
            data : torch.Tensor
                Input tensor of shape (1, 1, H, W).

            Returns
            -------
            float
                Mean of the activated output.
            """
            with torch.no_grad():
                out = self.forward(data)
                return out.mean().item()

    return ConvFilter()
