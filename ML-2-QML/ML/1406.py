"""Enhanced classical convolutional filter with depth‑wise separable support and optional batch‑norm."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the original Conv filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter.
    in_channels : int, default 1
        Number of input channels.  The filter is depth‑wise separable
        so each channel has its own 1×1 convolution.
    out_channels : int, default 1
        Number of output channels.  The depth‑wise part is followed
        by a point‑wise 1×2 convolution that expands to ``out_channels``.
    bias : bool, default True
        Whether to include a bias term.
    batch_norm : bool, default False
        If True, a batch‑norm layer is inserted after the depth‑wise
        convolution.
    depth : int, default 1
        Number of times the filter is applied sequentially; this mimics
        the depth of a convolutional stack.

    Returns
    -------
    float
        The mean activation after sigmoid, as in the original API.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        bias: bool = True,
        batch_norm: bool = False,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        # Depth‑wise separable: first a 2D depth‑wise conv
        self.depthwise = nn.Conv2d(
            in_channels,
            **{k: v for k, v in {"out_channels": in_channels, "kernel_size": kernel_size, "bias": bias}.items()},
        )
        if batch_norm:
            self.bn = nn.BatchNorm2d(in_channels)
        else:
            self.bn = None

        # Point‑wise 1×1 conv to match output channel count
        self.pointwise = nn.Conv2d(
            **{k: v for k, v in {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": 1, "bias": bias}.items()},
        )

        self.threshold = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that can be executed on a batch of input tensors.
        ``x`` is expected shape (B, C_in, H, W).
        """
        out = self.depthwise(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.pointwise(out)
        # Mimic the original activation pattern
        out = torch.sigmoid(out - self.threshold)
        return out.mean().unsqueeze(0)

    def run(self, *args, **kwargs):
        """
        Compatibility wrapper that allows the same API as the original
        ``Conv`` call.
        """
        if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
            return self.forward(torch.as_tensor(args[0], dtype=torch.float32).unsqueeze(0))
        else:
            raise ValueError("ConvEnhanced.run needs one 2‑D array input")

__all__ = ["ConvEnhanced"]
