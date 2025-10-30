from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ConvFilter(nn.Module):
    """Classic 2‑D convolution filter that emulates the quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data expected shape (batch, 1, H, W) or (1, H, W)
        x = self.conv(data)
        x = torch.sigmoid(x - self.threshold)
        return x

class ConvGen259(nn.Module):
    """Classical convolutional network inspired by quantum circuits.

    Builds on a basic ConvFilter and a stack of fully‑connected
    layers that emulate the structure of a QCNN.  The network
    accepts a single 2‑D image patch and produces a binary
    probability.  It is fully differentiable and can be used
    as a drop‑in replacement for the quantum implementation
    in downstream pipelines.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        num_features: int = 8,
        hidden_sizes: tuple[int,...] = (16, 12, 8, 4, 4)
    ) -> None:
        super().__init__()
        self.filter = ConvFilter(kernel_size, threshold)

        # Feature‑map + convolution‑pool stages
        layers = []
        for out_ch in hidden_sizes:
            layers.append(nn.Linear(num_features, out_ch))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Tensor of shape (batch, 1, H, W) or (1, H, W).

        Returns:
            Tensor of shape (batch, 2) with class probabilities.
        """
        x = self.filter(inputs)
        # Flatten to (batch, num_features)
        x = torch.flatten(x, 1)
        logits = self.fc(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["ConvGen259"]
