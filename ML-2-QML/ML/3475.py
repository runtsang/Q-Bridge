"""
Hybrid classical-quantum estimator that combines a convolutional feature extractor
with a fully‑connected regression head.

The design mirrors the original EstimatorQNN example but replaces the
single‑layer feed‑forward architecture with a convolutional filter
(`ConvFilter`) followed by a shallow neural network.  The
`ConvFilter` is inspired by the `Conv` class in the second reference pair.
This hybrid layout allows easy experimentation with different feature
extraction strategies while retaining a pure PyTorch implementation.
"""

from __future__ import annotations

import torch
from torch import nn


def ConvFilter(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    class _ConvFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()
    return _ConvFilter()


class HybridEstimatorQNN(nn.Module):
    """Classical hybrid estimator: ConvFilter + linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv_filter = ConvFilter()
        self.network = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Expect inputs to be a 2D array (kernel_size, kernel_size)
        conv_output = self.conv_filter.run(inputs)
        x = torch.tensor([conv_output], dtype=torch.float32)
        return self.network(x)


def EstimatorQNN() -> HybridEstimatorQNN:
    """Return an instance of the hybrid classical estimator."""
    return HybridEstimatorQNN()


__all__ = ["EstimatorQNN"]
