"""Combined classical convolution and sampler network.

This module defines ConvGen214, a hybrid model that first applies a
convolutional filter to a 2‑D input and then feeds the scalar result
into a small neural sampler.  The class can be used as a drop‑in
replacement for the original Conv() function while providing a richer
output space.

The implementation keeps the original API surface: Conv() returns an
instance of ConvGen214.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def Conv():
    """Return a ConvGen214 instance for compatibility with the original API."""
    return ConvGen214()


class ConvFilter(nn.Module):
    """Classical convolutional filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        """Apply the filter to a 2‑D array and return a scalar."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class SamplerModule(nn.Module):
    """Small sampler network producing a probability distribution."""

    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class ConvGen214(nn.Module):
    """Hybrid model combining a convolutional filter and a sampler network."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, hidden: int = 4) -> None:
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size, threshold)
        self.sampler = SamplerModule(hidden)

    def forward(self, data) -> torch.Tensor:
        """Run the full pipeline and return the sampler output."""
        conv_value = self.conv_filter.run(data)
        conv_tensor = torch.tensor([conv_value], dtype=torch.float32)
        return self.sampler(conv_tensor)

    def run(self, data) -> torch.Tensor:
        """Convenience wrapper around forward."""
        return self.forward(data)


__all__ = ["ConvGen214", "Conv"]
