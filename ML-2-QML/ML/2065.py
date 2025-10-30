"""Enhanced classical convolutional filter with channel support and adaptive threshold."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class ConvEnhanced(nn.Module):
    """
    Drop‑in replacement for the original Conv filter that supports multiple
    input/output channels, configurable kernel sizes, and an optional learnable
    threshold that adapts during training.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int | tuple[int, int] = 2,
        threshold: float | None = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )
        self.threshold = (
            nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
            if threshold is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution, sigmoid activation and optional threshold gating.
        """
        logits = self.conv(x)
        if self.threshold is None:
            activations = torch.sigmoid(logits)
        else:
            activations = torch.sigmoid(logits - self.threshold)
        return activations

    def compute_mean_activation(self, x: torch.Tensor) -> float:
        """
        Return the mean of all activations across batch, channel, and spatial dimensions.
        """
        return self(x).mean().item()

__all__ = ["ConvEnhanced"]
