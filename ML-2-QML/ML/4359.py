"""Hybrid classical convolutional filter with quantum-inspired gating and pooling.

This module extends the original Conv.py by combining ideas from
QCNN (multi‑stage pooling), FraudDetection (parameter clipping and
activation scaling), and QCNet (quantum‑style hybrid head).  The class
`ConvGen034` can be used as a drop‑in replacement for the original
`Conv()` factory while providing a richer feature extraction pipeline.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
import math


@dataclass
class ConvParams:
    """Container for the configuration of the hybrid filter."""
    kernel_size: int = 3
    threshold: float = 0.0
    pooling: bool = True
    bias: bool = True
    clip: bool = True
    clip_bound: float = 5.0


class ConvGen034(nn.Module):
    """Hybrid convolutional filter with optional pooling and clipped
    activation scaling.

    The forward pass mimics a quantum measurement by applying a sigmoid
    gate after a linear threshold.  When `pooling` is enabled, a
    MaxPool2d layer is applied, emulating the pooling stages of QCNN.
    The optional `clip` flag constrains the linear weights and bias,
    inspired by the FraudDetection photonic layers.
    """

    def __init__(self, params: ConvParams | None = None) -> None:
        super().__init__()
        if params is None:
            params = ConvParams()
        self.params = params
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.params.kernel_size,
            bias=self.params.bias,
        )
        if self.params.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = None

        if self.params.clip:
            # Clip weights and bias to a bounded interval
            with torch.no_grad():
                self.conv.weight.data.clamp_(
                    -self.params.clip_bound, self.params.clip_bound
                )
                if self.conv.bias is not None:
                    self.conv.bias.data.clamp_(
                        -self.params.clip_bound, self.params.clip_bound
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape (batch, 1, H, W)
        x = self.conv(x)
        # Quantum‑style sigmoid after threshold
        x = torch.sigmoid(x - self.params.threshold)
        if self.pool is not None:
            x = self.pool(x)
        # Return the mean activation as a scalar per sample
        return x.mean(dim=(1, 2, 3))

def Conv() -> ConvGen034:
    """Factory returning a pre‑configured ConvGen034 instance."""
    return ConvGen034()

__all__ = ["ConvGen034", "Conv", "ConvParams"]
