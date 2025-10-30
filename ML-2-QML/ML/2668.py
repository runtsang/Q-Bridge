"""Hybrid sampler network combining a classical convolutional filter with a quantum-inspired sampler."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """A lightweight 2‑D convolutional filter that mimics a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the filter and return a 2‑element feature vector."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        mean = activations.mean()
        # Duplicate the scalar to produce a 2‑D feature vector
        return torch.stack([mean, mean])

class HybridSamplerQNN(nn.Module):
    """Hybrid classical sampler that first extracts features with ConvFilter
    and then maps them through a small feed‑forward network."""
    def __init__(self, conv_kernel: int = 2, conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass: conv → feature vector → sampler network → softmax probabilities."""
        features = self.conv.run(inputs)
        logits = self.net(features)
        return F.softmax(logits, dim=-1)

def SamplerQNN() -> HybridSamplerQNN:
    """Return a ready‑to‑use hybrid sampler module."""
    return HybridSamplerQNN()

__all__ = ["SamplerQNN"]
