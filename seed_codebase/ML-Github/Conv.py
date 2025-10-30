"""Classical convolutional filter used as a drop-in replacement for quanvolution."""

from __future__ import annotations

import torch
from torch import nn


def Conv():
    """Return a callable object that emulates the quantum filter with PyTorch ops."""

    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
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

    return ConvFilter()


__all__ = ["Conv"]
