"""Hybrid classical convolution + sampler network inspired by Conv.py and SamplerQNN.py."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Convolutional filter emulating a quantum filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Return mean activation after sigmoid threshold."""
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

class SamplerModule(nn.Module):
    """Simple feed‑forward sampler network."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class ConvGen133(nn.Module):
    """Hybrid classical model combining a convolution filter and a sampler network."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size, threshold)
        self.sampler = SamplerModule()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the convolution filter then feed its scalar output to the sampler."""
        activation = self.conv(data)
        vec = torch.stack([activation, activation], dim=-1)
        return self.sampler(vec)

__all__ = ["ConvGen133"]
