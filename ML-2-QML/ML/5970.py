"""Hybrid convolution–sampler architecture combining classical convolutional filtering with a neural sampler.

The module exposes a single class ``HybridConvSampler`` that can be instantiated as a drop‑in
replacement for either the original Conv or SamplerQNN modules.  It
leverages a learnable 2‑D convolution to extract local features and a
small feed‑forward network to produce class probabilities.  The design
mirrors the two reference seeds while providing a unified API
compatible with the anchor ``Conv.py``.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Learnable 2‑D convolution filter with a sigmoid activation."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

class SamplerModule(nn.Module):
    """Small feed‑forward sampler that maps 2‑D features to a probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class HybridConvSampler(nn.Module):
    """Hybrid model that chains a convolutional filter with a sampler network.

    Args:
        kernel_size: Size of the convolution kernel.
        threshold: Activation threshold for the ConvFilter.
        sampler: Instance of ``SamplerModule``.  If ``None`` a default sampler is used.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 sampler: nn.Module | None = None) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=kernel_size, threshold=threshold)
        self.sampler = sampler if sampler is not None else SamplerModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape ``(B, 1, H, W)`` where ``B`` is batch size.

        Returns
        -------
        torch.Tensor
            Class probability tensor of shape ``(B, 2)``.
        """
        # Convolution → sigmoid → mean over spatial dims → flatten
        conv_out = self.conv(x)
        # Global mean pooling
        pooled = conv_out.mean(dim=[2, 3])
        # Ensure 2‑D input for sampler
        features = pooled.view(pooled.size(0), -1)
        return self.sampler(features)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class index with highest probability."""
        probs = self.forward(x)
        return probs.argmax(dim=-1)

__all__ = ["HybridConvSampler"]
