"""ConvEnhanced: classical depthwise separable convolution with multi‑scale kernels and adaptive threshold."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import List


class ConvEnhanced(nn.Module):
    """Drop‑in replacement for the original Conv filter.

    Combines depthwise separable convolution, multiple kernel sizes (e.g. 1×1, 2×2, 3×3),
    and a learnable threshold that is updated during training. The public API mirrors
    the seed: ``run`` accepts a 2‑D array and returns a scalar feature value.
    """

    def __init__(
        self,
        kernel_sizes: List[int] = [1, 2, 3],
        in_channels: int = 1,
        out_channels: int = 1,
        depthwise: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise = depthwise

        # Depthwise separable convolution for each kernel size
        self.depthwise_convs = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()
        for k in kernel_sizes:
            # depthwise
            dw = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=k,
                groups=in_channels,
                bias=bias,
                padding=k // 2,
            )
            self.depthwise_convs.append(dw)
            # pointwise
            pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            self.pointwise_convs.append(pw)

        # Learnable threshold
        self.threshold = nn.Parameter(torch.zeros(1))

    def run(self, data) -> float:
        """Compute the scalar feature for a 2‑D input array."""
        if isinstance(data, np.ndarray):
            x = torch.from_numpy(data.astype(np.float32))
        else:
            x = data.clone().detach()

        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (B, C, H, W)
        elif x.ndim == 3:
            x = x.unsqueeze(0)  # (B, H, W)
        else:
            raise ValueError("Input must be 2‑D or 3‑D array")

        outputs = []
        for dw, pw in zip(self.depthwise_convs, self.pointwise_convs):
            y = dw(x)
            y = pw(y)
            y = torch.sigmoid(y - self.threshold)
            outputs.append(y)

        # Aggregate across scales (mean)
        out = torch.mean(torch.stack(outputs), dim=0)
        return out.mean().item()

    def forward(self, x):
        return self.run(x)


def Conv(*args, **kwargs):
    """Factory that returns a ConvEnhanced instance, keeping the original API."""
    return ConvEnhanced(*args, **kwargs)


__all__ = ["ConvEnhanced", "Conv"]
