"""Classical implementation of ConvFilter using depth‑wise separable convolution.

This module defines a drop‑in replacement for the original Conv class.
It offers a learnable threshold network and a `run` method that
produces a scalar activation probability.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

__all__ = ["ConvFilter"]


class ConvFilter(nn.Module):
    """
    Classical depth‑wise separable convolution filter with a learnable threshold.
    The filter outputs a single scalar in [0, 1] representing the probability
    of activation.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel (assumes square kernel).
    threshold : float, default 0.0
        Initial static threshold; overridden by the dynamic network.
    device : str or torch.device, default "cpu"
        Device on which to allocate tensors.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.device = device

        # Depth‑wise convolution: 1 input channel → 1 output channel
        self.depthwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
            padding=0,
        )

        # Small network that produces a dynamic threshold from the conv output
        self.threshold_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(kernel_size * kernel_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (1, H, W) or (N, 1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar activation probability in [0, 1].
        """
        if x.ndim == 3:
            # Add batch dimension
            x = x.unsqueeze(0)

        # Apply depth‑wise convolution
        conv_out = self.depthwise(x)

        # Compute dynamic threshold
        dyn_thresh = self.threshold_net(conv_out).view(-1, 1, 1, 1)

        # Apply sigmoid with threshold
        activations = torch.sigmoid(conv_out - dyn_thresh)

        # Return mean activation probability
        return activations.mean()

    def run(self, data: torch.Tensor | Tuple[int, int, int] | list | tuple) -> float:
        """
        Convenience method that accepts raw data and returns a float.

        Parameters
        ----------
        data : torch.Tensor or array‑like
            Input image of shape (H, W) or (1, H, W).

        Returns
        -------
        float
            Activation probability.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
        return self.forward(data).item()
