"""Hybrid classical sampler network combining convolutional feature extraction and a softmax sampler.

This module defines a `SamplerQNN` class that first applies a 2×2 convolutional filter to
image-like inputs, then flattens the feature map and passes it through a small feed‑forward
network to produce a probability distribution over two classes.  The architecture is
inspired by the `QuanvolutionFilter` from the Quanvolution example and the original
`SamplerQNN` feed‑forward sampler.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """Classical sampler network with a convolutional front‑end and a softmax sampler."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor: 1→4 channels, 2×2 kernel, stride 2
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # After conv the feature map has shape (batch, 4, 14, 14) for 28×28 input
        # Flatten and feed into a two‑layer MLP
        self.fc1 = nn.Linear(4 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 1, H, W). Expected 28×28 images.

        Returns:
            Tensor of shape (batch, 2) containing class probabilities.
        """
        features = self.conv(x)          # (batch, 4, 14, 14)
        flat = features.view(features.size(0), -1)  # (batch, 4*14*14)
        hidden = F.relu(self.fc1(flat))
        logits = self.fc2(hidden)
        return F.softmax(logits, dim=-1)


def SamplerQNN() -> SamplerQNN:
    """Factory returning a freshly instantiated `SamplerQNN`."""
    return SamplerQNN()


__all__ = ["SamplerQNN"]
