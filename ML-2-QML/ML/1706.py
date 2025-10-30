"""Enhanced classical convolutional filter.

Drop‑in replacement for the original Conv filter.
Supports multi‑scale kernels, learnable threshold, and optional
self‑attention over filter responses.

The interface mirrors the seed: an instance exposes a `run` method
that accepts a 2‑D array and returns a float.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Union

class ConvEnhanced(nn.Module):
    """
    Multi‑scale, attention‑aware convolutional filter.

    Parameters
    ----------
    kernel_size
        Either an int or an iterable of ints specifying the filter size(s).
    in_channels, out_channels
        Convolution channel dimensions.
    bias
        Whether to use a bias term.
    attention
        If True, a learned attention weighting is applied to the
        spatial average of each output channel before pooling.
    """
    def __init__(
        self,
        kernel_size: Union[int, Iterable[int]] = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        bias: bool = True,
        attention: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_sizes = [kernel_size]
        else:
            self.kernel_sizes = list(kernel_size)

        self.convs = nn.ModuleList(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=k,
                stride=1,
                padding=0,
                bias=bias,
            )
            for k in self.kernel_sizes
        )
        self.threshold = nn.Parameter(torch.zeros(1))
        self.attention = attention
        if attention:
            self.attn = nn.Linear(out_channels, out_channels)

    def _apply_kernel(self, data: torch.Tensor) -> torch.Tensor:
        """Apply each convolution and stack along a new dimension."""
        outs = []
        for conv in self.convs:
            outs.append(conv(data))
        return torch.stack(outs, dim=0)

    def forward(self, data) -> float:
        """
        Run the filter on the input data.

        Parameters
        ----------
        data
            2‑D array or torch.Tensor of shape (H, W) or (C, H, W).

        Returns
        -------
        float
            Averaged, sigmoid‑activated, optionally attention‑weighted
            response.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # batch, channel, H, W
        elif data.ndim == 3:
            data = data.unsqueeze(0)  # batch, channel, H, W

        out = self._apply_kernel(data)
        out = torch.sigmoid(out - self.threshold)

        # Aggregate over kernel‑size dimension
        out = out.mean(dim=0)  # shape: (batch, out_channels, H, W)

        if self.attention:
            pooled = out.mean(dim=(2, 3))  # (batch, out_channels)
            attn_weights = torch.softmax(self.attn(pooled), dim=1)
            out = (out * attn_weights.unsqueeze(-1).unsqueeze(-1)).mean()

        return out.mean().item()

    def run(self, data):
        return self.forward(data)

__all__ = ["ConvEnhanced"]
