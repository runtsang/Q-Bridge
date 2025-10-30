"""Enhanced classical convolutional filter with optional batch‑norm and auto‑tuning."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable, Tuple, Optional

__all__ = ["ConvEnhanced"]


class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the legacy Conv filter that adds:

    * Learnable 2‑D convolutional kernel (size 2‑8, stride 1).
    * Optional batch‑normalization.
    * Auto‑tuning over a small grid of kernel sizes and thresholds.
    * A ``run`` method that accepts a NumPy array or torch tensor and
      returns a scalar probability‑like score.
    """

    def __init__(
        self,
        kernel_size: int = 4,
        threshold: float = 0.0,
        batch_norm: bool = False,
        auto_tune: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.batch_norm = batch_norm
        self.auto_tune = auto_tune

        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, bias=True
        )
        if batch_norm:
            self.bn = nn.BatchNorm2d(1)

        if auto_tune:
            self._auto_tune()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return x

    def forward(
        self, data: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """
        Forward pass that accepts a 2‑D array or torch tensor.
        The input is reshaped to match the convolutional kernel.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))
        if data.ndim!= 2:
            raise ValueError("Input must be a 2‑D array.")
        x = data.reshape(1, 1, self.kernel_size, self.kernel_size)
        return self._forward(x)

    def run(self, data: torch.Tensor | np.ndarray) -> float:
        """
        Run a single example through the filter and return the mean activation.
        """
        return self.forward(data).mean().item()

    def _auto_tune(self) -> None:
        """
        Simple grid search over kernel sizes and thresholds.
        The best pair is chosen based on the highest mean activation.
        """
        best_cfg: Tuple[int, float] = (self.kernel_size, self.threshold)
        best_val = -float("inf")

        for k in [2, 4, 6, 8]:
            for t in [0.0, 0.5, 1.0]:
                conv = nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=k,
                    stride=1,
                    bias=True,
                )
                if self.batch_norm:
                    bn = nn.BatchNorm2d(1)
                else:
                    bn = None

                # Create a dummy dataset of 5 random patches
                dummy = torch.rand(5, 1, k, k)
                activations = []
                for patch in dummy:
                    x = conv(patch)
                    if bn is not None:
                        x = bn(x)
                    activations.append(x.mean().item())

                val = max(activations)
                if val > best_val:
                    best_val = val
                    best_cfg = (k, t)

        self.kernel_size, self.threshold = best_cfg
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=1,
            bias=True,
        )
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(1)
