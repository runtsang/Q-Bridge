"""ConvGen315 – a depth‑wise separable convolution with a learnable threshold and batch support.

This module extends the original Conv filter by:
- Supporting multiple depth‑wise separable kernels.
- Exposing a learnable threshold parameter.
- Accepting batched inputs and returning a tensor of activations.

The class can be used as a drop‑in replacement for Conv.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import List, Optional

class ConvGen315(nn.Module):
    """Depth‑wise separable convolution with learnable threshold."""
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Optional[List[int]] = None,
        threshold: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes or [2]
        # Depth‑wise separable convolutions: one filter per input channel
        self.depthwise = nn.ModuleList()
        for k in self.kernel_sizes:
            self.depthwise.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    groups=in_channels,
                    bias=bias,
                )
            )
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, in_channels, H, W)

        Returns:
            Tensor of shape (batch,) containing mean activation per sample.
        """
        activations = []
        for conv in self.depthwise:
            out = conv(x)  # shape: (batch, out_channels, H', W')
            out = self.activation(out - self.threshold)
            activations.append(out.mean(dim=[1, 2, 3]))
        # Sum activations from all kernels
        return torch.stack(activations, dim=-1).mean(dim=-1)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """Compatibility wrapper for the original API."""
        return self.forward(data)

__all__ = ["ConvGen315"]
