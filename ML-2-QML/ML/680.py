"""Hybrid classical convolutional filter with adaptive threshold and depth control.

This module defines ConvEnhanced, a drop‑in replacement for the original
Conv class.  It implements a stack of 2‑D convolutional layers that
optionally use a dynamic threshold function.  The forward pass
returns the mean activation across the channel, matching the API of
the original Conv filter.

The class can be used in a standard PyTorch training loop and
supports back‑propagation.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Callable, Optional

__all__ = ["ConvEnhanced"]


class ConvEnhanced(nn.Module):
    """
    Multi‑layer 2‑D convolutional filter with optional adaptive threshold.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        depth: int = 1,
        threshold: float = 0.0,
        threshold_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int, default 2
            Size of the convolutional kernel.
        depth : int, default 1
            Number of convolutional layers to stack.
        threshold : float, default 0.0
            Static threshold applied after each convolution.
        threshold_func : callable, optional
            Function that receives the output of a convolution and returns
            a threshold tensor of the same shape.  If provided, it overrides
            the static threshold for that layer.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.depth = depth
        self.threshold = threshold
        self.threshold_func = threshold_func

        # Create a list of convolutional layers
        self.convs: nn.ModuleList = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the filter.

        Parameters
        ----------
        data : torch.Tensor
            2‑D input image of shape (H, W).  It is internally reshaped to
            (1, 1, H, W) for the convolutional layers.

        Returns
        -------
        torch.Tensor
            Mean activation value after the last convolutional layer.
        """
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        else:
            raise ValueError("Input data must be a 2‑D array.")
        x = data
        for conv in self.convs:
            logits = conv(x)
            # Determine threshold for this layer
            if self.threshold_func is not None:
                th = self.threshold_func(logits)
            else:
                th = torch.full_like(logits, self.threshold)
            activations = torch.sigmoid(logits - th)
            x = activations
        # Return mean activation value
        return x.mean()

    def run(self, data) -> float:
        """
        Convenience wrapper to run the filter on a NumPy array and
        return a Python float, mirroring the original Conv API.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        mean_activation = self.forward(tensor)
        return mean_activation.item()
