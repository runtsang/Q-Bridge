"""Hybrid classical convolutional filter that merges a trainable 2x2 kernel with a quantum-inspired measurement."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["ConvGen025"]

class ConvGen025(nn.Module):
    """
    Classical convolutional filter with a quantum-inspired path.

    The filter can be used as a drop-in replacement for the original Conv() function.
    It implements a trainable 2x2 convolution and a simulated quantum measurement
    that outputs a probability-like score for each patch. The two outputs are
    combined into a single scalar.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)

        # Quantum-inspired parameters
        self.quantum_weights = nn.Parameter(torch.randn(kernel_size * kernel_size))

    def run(self, data) -> float:
        """
        Run the filter on a 2D array and return a scalar activation.

        Args:
            data: 2D array of shape (kernel_size, kernel_size).

        Returns:
            float: combined classical and quantum activation.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        classical = torch.sigmoid(logits - self.threshold)

        # Quantum-inspired part
        flattened = tensor.view(1, -1)
        weighted = torch.dot(flattened.squeeze(), self.quantum_weights)
        quantum = torch.sigmoid(weighted - self.threshold)

        return (classical.mean() + quantum).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batched input.

        Args:
            x: Tensor of shape (N, 1, H, W).

        Returns:
            Tensor of shape (N, 1, H', W') after classical convolution.
        """
        return self.conv(x)
