"""Enhanced classical convolutional filter with trainable weights, dropout, and bias handling.

The class mimics a quantum filter but offers richer functionality:
- Trainable convolutional kernel (torch.nn.Conv2d)
- Optional dropout for regularisation
- Flexible thresholding applied after sigmoid
- Batch support via forward method

The interface remains compatible with the original `Conv()` helper: calling `Conv()` returns an instance with a `run` method that accepts a 2‑D array.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def Conv() -> nn.Module:
    """Return a callable object that emulates a quantum filter with richer classical ops."""

    class ConvFilter(nn.Module):
        def __init__(
            self,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            threshold: float = 0.0,
            dropout: float | None = 0.0,
            bias: bool = True,
        ) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.threshold = threshold
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
            self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

            # initialise weights to approximate a simple edge detector
            nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
            if bias:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.conv.bias, -bound, bound)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Standard forward pass for a 4‑D batch tensor."""
            out = self.conv(x)
            out = torch.sigmoid(out - self.threshold)
            out = self.dropout(out)
            return out

        def run(self, data: torch.Tensor | list | tuple) -> float:
            """Convenience wrapper to keep the original API.

            Accepts a 2‑D array, applies the filter and returns the mean activation.
            """
            tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            logits = self.forward(tensor)
            return logits.mean().item()

    return ConvFilter()
