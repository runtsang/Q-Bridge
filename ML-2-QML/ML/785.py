"""Hybrid convolutional filter with a learnable bias and dropout for classical training."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def ConvGen050():
    """Return a callable class that extends the original Conv filter.

    The class exposes a ``run`` method that accepts a 2×2 patch and returns a
    scalar activation.  It can be used as a drop‑in replacement for the
    original ``Conv`` in any pipeline that expects a single‑output
    convolution.  The implementation now supports a learnable bias,
    dropout regularisation and a small 2×2 kernel that is trained
    jointly with the rest of the network.
    """

    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                     dropout: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.dropout = dropout
            # 2×2 kernel with a bias that can be trained
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
            # Dropout to regularise the activations
            self.drop = nn.Dropout2d(p=dropout)

        def forward(self, patch: torch.Tensor) -> torch.Tensor:
            """Forward pass that returns a scalar activation for a 2×2 patch.

            The patch is expected to be a 2‑D tensor of shape (kernel_size,
            kernel_size).  The method reshapes it to the batch‑size‑1
            format required by ``nn.Conv2d`` and then applies the
            convolution, sigmoid activation, and optional dropout.
            """
            # Ensure a batch dimension
            patch = patch.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
            logits = self.conv(patch)
            logits = self.drop(logits)
            activations = torch.sigmoid(logits - self.threshold)
            # Return a scalar value
            return activations.mean()

        def run(self, data) -> float:
            """Convenience wrapper that accepts a NumPy array and returns a float."""
            tensor = torch.as_tensor(data, dtype=torch.float32)
            return self.forward(tensor).item()

    return ConvFilter()
