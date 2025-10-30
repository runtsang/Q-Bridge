"""Enhanced classical convolutional filter with multi‑scale depth‑wise separable kernels
and a learnable threshold.

The class can be used as a drop‑in replacement for the original Conv module.
It accepts a 2‑D array and supports multiple kernel sizes in a single forward pass.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Sequence

class ConvFilter(nn.Module):
    """
    Parameters
    ----------
    kernel_sizes : Iterable[int]
        List of kernel sizes (e.g., [3, 5, 7]) to create a number of
        depth‑wise separable convolutions.
    stride : int, optional
        Stride of the convolution. Default is 1.
    padding : int, optional
        Zero‑padding added to both sides of the input. Default is 0.
    threshold : float, optional
        Initial threshold for the sigmoid activation. It is registered as a
        learnable parameter.
    """

    def __init__(
        self,
        kernel_sizes: Iterable[int] = (3,),
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.stride = stride
        self.padding = padding

        # Create a depth‑wise separable conv for each kernel size.
        self.convs = nn.ModuleList()
        for k in self.kernel_sizes:
            # depth‑wise conv (groups=1 because we only have one input channel)
            depth = nn.Conv2d(1, 1, kernel_size=k, stride=stride, padding=padding, bias=True)
            # point‑wise conv to keep number of channels unchanged
            point = nn.Conv2d(1, 1, kernel_size=1, bias=True)
            self.convs.append(nn.Sequential(depth, point))

        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C=1, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, 1, H', W') where H', W' depend on the
            convolution parameters.
        """
        outs = []
        for conv in self.convs:
            out = conv(x)
            out = torch.sigmoid(out - self.threshold)
            outs.append(out)
        # Aggregate over kernel sizes by averaging
        return torch.mean(torch.stack(outs, dim=0), dim=0)

    def run(self, data: Sequence[Sequence[float]]) -> float:
        """
        Convenience method for a single 2‑D array.

        Parameters
        ----------
        data : 2‑D sequence of floats

        Returns
        -------
        float
            Mean activation value after the convolution and sigmoid.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(tensor)
        return out.mean().item()

__all__ = ["ConvFilter"]
