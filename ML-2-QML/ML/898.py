"""ConvGen: multi‑scale, self‑normalizing classical convolutional filter.

This module provides a drop‑in replacement for the original Conv() factory. It supports
multiple kernel sizes, optional depth‑wise separable convolutions, batch‑norm, and
self‑normalizing activations. The forward pass returns a mean activation score
per input sample, matching the interface of the original ConvFilter.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, List, Tuple


class ConvGen(nn.Module):
    """
    Multi‑scale convolutional filter with optional depth‑wise separable
    convolutions and batch‑norm. The module is fully differentiable and
    can be inserted into a larger neural network.
    """

    def __init__(
        self,
        kernel_sizes: Iterable[int] | Tuple[int,...] = (2, 4, 8),
        depthwise: bool = True,
        norm: bool = True,
        threshold: float = 0.0,
        bias: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        kernel_sizes : iterable of int
            Kernel sizes to use. The first element is the shallowest filter.
        depthwise : bool
            If True, use depth‑wise separable convolutions.
        norm : bool
            If True, apply BatchNorm2d after each convolution.
        threshold : float
            Activation threshold applied before sigmoid.
        bias : bool
            Whether convolution layers include a bias term.
        """
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.threshold = threshold
        self.convs: List[nn.Module] = nn.ModuleList()

        for k in self.kernel_sizes:
            if depthwise:
                conv = nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=k, groups=1, bias=bias),
                    nn.BatchNorm2d(1) if norm else nn.Identity(),
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=k, bias=bias),
                    nn.BatchNorm2d(1) if norm else nn.Identity(),
                )
            self.convs.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns a mean activation score per sample.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch,) containing the mean activation per sample.
        """
        scores: List[torch.Tensor] = []

        for conv in self.convs:
            out = conv(x)  # shape (batch, 1, H-k+1, W-k+1)
            out = torch.sigmoid(out - self.threshold)
            # Mean over spatial dimensions
            scores.append(out.mean(dim=[2, 3]))

        # Stack scores: shape (num_kernels, batch)
        stacked = torch.stack(scores, dim=0)  # (K, batch)
        # Simple average over kernel sizes
        return stacked.mean(dim=0).squeeze()

    def run(self, data: torch.Tensor | List[List[float]]) -> float:
        """
        Convenience wrapper that accepts a 2‑D array and returns a scalar
        activation score, matching the interface of the original ConvFilter.

        Parameters
        ----------
        data : torch.Tensor or list of lists
            Input data of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean activation score.
        """
        if isinstance(data, list):
            data = torch.tensor(data, dtype=torch.float32)
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return self.forward(data).item()


# Public API
__all__ = ["ConvGen"]
