"""ConvGen351: a multi‑scale, residual, depthwise‑separable convolutional filter.

This module implements a drop‑in replacement for the original Conv filter while adding richer functionality:
* Multi‑scale kernels (1×1, 3×3, 5×5) with learnable weights.
* Residual shortcut that adds the input to the output of the convolutional branch.
* Depthwise separable convolution to keep the number of parameters small.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ConvGen351(nn.Module):
    """Drop‑in replacement for the original Conv filter.

    The forward pass follows the same ``run`` interface used in the seed code.
    """

    def __init__(
        self,
        kernel_sizes: tuple[int, int, int] = (1, 3, 5),
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.threshold = threshold

        # Depthwise separable convolution: depthwise conv + pointwise conv
        self.depthwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_sizes[0],
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

        # Residual shortcut
        self.residual = nn.Identity()

        # Multi‑scale branch (optional)
        self.multi_scale = nn.ModuleList()
        for ks in kernel_sizes[1:]:
            self.multi_scale.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=ks,
                    stride=stride,
                    padding=ks // 2,
                    bias=True,
                )
            )

    def run(self, data) -> float:
        """Run the classical filter on 2‑D data.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).

        Returns:
            float: average sigmoid activation after convolution and residual addition.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        dw = self.depthwise(tensor)
        pw = self.pointwise(dw)
        out = pw + self.residual(tensor)

        # Add multi‑scale contributions
        for conv in self.multi_scale:
            out += conv(tensor)

        activations = torch.sigmoid(out - self.threshold)
        return activations.mean().item()
