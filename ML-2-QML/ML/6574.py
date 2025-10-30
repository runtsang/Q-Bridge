"""Hybrid classical convolutional sampler.

This module provides a drop‑in replacement for the original ``Conv`` function
while adding a lightweight sampler network.  The implementation is fully
classical and relies only on PyTorch.

The class ``HybridConvSampler`` exposes a convolutional filter followed by a
two‑layer MLP that outputs class probabilities.  It can be instantiated
directly or via the ``Conv`` helper.

Example
-------
>>> from Conv__gen418 import Conv
>>> model = Conv()
>>> probs = model(torch.randn(1, 1, 2, 2))
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HybridConvSampler(nn.Module):
    """Classical hybrid conv + sampler network."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 sampler_hidden: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Sampler network: small MLP producing 2‑class probabilities
        self.sampler = nn.Sequential(
            nn.Linear(kernel_size * kernel_size, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (batch, 2).
        """
        # Convolution and sigmoid activation
        conv_out = self.conv(x)
        conv_out = torch.sigmoid(conv_out - self.threshold)

        # Flatten and feed into sampler
        flat = conv_out.view(conv_out.size(0), -1)
        logits = self.sampler(flat)
        return F.softmax(logits, dim=-1)


def Conv() -> HybridConvSampler:
    """
    Return a callable object that emulates the quantum filter with PyTorch ops.
    This wrapper preserves API compatibility with the original ``Conv`` function.
    """
    return HybridConvSampler()


__all__ = ["Conv", "HybridConvSampler"]
