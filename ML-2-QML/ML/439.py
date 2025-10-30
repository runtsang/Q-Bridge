"""Enhanced classical convolutional filter with residual connections and batch‑normalization.

The Conv() factory returns a ConvFilter instance that can be used as a drop‑in
replacement for the original quantum filter.  The network supports multiple
convolutional layers, configurable kernel sizes, optional residual
connections, and batch‑norm.  A small ``run`` method is retained for
compatibility with the original API: it accepts a 2‑D array, runs it through
the network and returns the mean activation as a float.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Sequence, Iterable


class ConvFilter(nn.Module):
    """
    Classical convolutional filter that can be used as a drop‑in replacement for
    the original quantum filter.

    Parameters
    ----------
    kernel_sizes : Sequence[int] | int, default=2
        Kernel sizes for each convolutional layer.  If an integer is provided,
        a single layer with that kernel size is created.
    depth : int, default=3
        Number of convolutional layers.
    use_residual : bool, default=True
        Whether to add a residual connection between the input and the output
        of each layer.
    threshold : float, default=0.0
        Threshold used in the final sigmoid activation.
    """

    def __init__(
        self,
        kernel_sizes: Sequence[int] | int = 2,
        depth: int = 3,
        use_residual: bool = True,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * depth
        assert len(kernel_sizes) == depth, "kernel_sizes length must match depth"

        self.depth = depth
        self.use_residual = use_residual
        self.threshold = threshold

        layers: list[nn.Module] = []
        in_channels = 1
        for k in kernel_sizes:
            conv = nn.Conv2d(in_channels, 1, kernel_size=k, bias=False)
            bn = nn.BatchNorm2d(1)
            layers.append(nn.Sequential(conv, bn, nn.ReLU(inplace=True)))
            in_channels = 1  # keep single channel throughout

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, 1, H, W) or a 2‑D array that will be
            reshaped to (1, 1, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after the final convolution and sigmoid activation.
        """
        if x.ndim == 2:
            x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4 and x.shape[1] == 1:
            x = x.float()
        else:
            raise ValueError("Input must be a 2‑D array or a (N,1,H,W) tensor.")

        out = x
        for layer in self.layers:
            residual = out if self.use_residual else 0
            out = layer(out) + residual

        out = torch.sigmoid(out - self.threshold)
        return out

    def run(self, data: Iterable[float] | torch.Tensor) -> float:
        """
        Compatibility wrapper that mimics the original API.

        Parameters
        ----------
        data : 2‑D array or tensor
            Input data to be processed by the filter.

        Returns
        -------
        float
            Mean activation of the filter output.
        """
        with torch.no_grad():
            out = self.forward(torch.as_tensor(data))
        return out.mean().item()


def Conv() -> ConvFilter:
    """Factory function that returns a ready‑to‑use ConvFilter instance."""
    return ConvFilter(kernel_sizes=[2] * 3, depth=3, use_residual=True, threshold=0.0)
