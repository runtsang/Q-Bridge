"""HybridSamplerQNN: Classical module combining convolution and sampling."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerQNN(nn.Module):
    """
    A hybrid sampler that first applies a 2×2 convolutional filter to the input
    and then feeds the resulting features into a small neural sampler network.
    The network is fully differentiable and can be trained together with a
    quantum backend via parameter‑to‑gradient hooks.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Classical convolutional filter (borrowed from Conv.py)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )

        # Classical sampler network (borrowed from SamplerQNN.py)
        self.sampler_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, H, W) containing grayscale patches.
        Returns:
            Tensor of shape (batch, 2) with softmax probabilities.
        """
        # Apply convolution
        conv_out = self.conv(x)  # shape (batch, 1, kernel, kernel)

        # Flatten to (batch, 2) – only the first two values are used
        conv_flat = conv_out.view(-1, 2)

        # Pass through sampler network
        logits = self.sampler_net(conv_flat)

        # Softmax over the two output logits
        probs = F.softmax(logits, dim=-1)
        return probs


__all__ = ["HybridSamplerQNN"]
