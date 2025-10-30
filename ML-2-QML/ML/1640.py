"""Classic depth‑wise separable quanvolution model with bias calibration.

The architecture mirrors the original quanvolution but replaces the single
convolution with a depth‑wise separable kernel and adds a learnable bias
term to mitigate class imbalance.  The model remains fully differentiable
and can be trained with standard optimizers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """Depth‑wise separable quanvolution with bias calibration."""

    def __init__(self) -> None:
        super().__init__()
        # Depth‑wise convolution: one filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            stride=2,
            groups=1,
            bias=False,
        )
        # Point‑wise 1×1 conv to expand to 4 feature maps
        self.pointwise = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 14 * 14, 10)
        # Learnable bias to correct class priors
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass for the classical quanvolution."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        logits = logits - self.bias  # bias calibration
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
