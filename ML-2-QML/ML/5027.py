"""
ML module for a hybrid classical‑quantum estimator.
Combines a classical convolution filter, a trainable self‑attention
layer, and a small regression network.  The design mirrors the
original EstimatorQNN but augments it with modern attention
mechanisms and a learnable convolutional front‑end, making it
more expressive while staying fully classical.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvFilter(nn.Module):
    """2‑D convolution filter with a sigmoid activation and a threshold."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mean activation over the output feature map."""
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean((2, 3), keepdim=True)


class SelfAttention(nn.Module):
    """Trainable self‑attention block operating on a batch of vectors."""

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rot = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.ent = nn.Parameter(torch.randn(embed_dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, embed_dim)
        Returns:
            Tensor of shape (batch, embed_dim)
        """
        query = x @ self.rot
        key = x @ torch.diag_embed(self.ent)
        scores = F.softmax(query @ key.transpose(-2, -1) / (self.embed_dim ** 0.5), dim=-1)
        return scores @ x


class EstimatorQNN(nn.Module):
    """Small regression network that can be used as a head."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridEstimatorQNN(nn.Module):
    """
    End‑to‑end hybrid model.
    1. ConvFilter processes the raw image.
    2. SelfAttention operates on the flattened convolution output.
    3. EstimatorQNN produces the final regression score.
    """

    def __init__(
        self,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        embed_dim: int = 4,
        hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(conv_kernel, conv_threshold)
        self.sa = SelfAttention(embed_dim)
        # The convolution output is a single scalar per sample;
        # we tile it to match the attention dimension.
        self.tile = nn.Linear(1, embed_dim, bias=False)
        self.estimator = EstimatorQNN(input_dim=embed_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, H, W)
        Returns:
            Tensor of shape (batch, 1)
        """
        conv_out = self.conv(x)          # (batch, 1, 1, 1)
        conv_out = conv_out.view(-1, 1)  # (batch, 1)
        attn_in = self.tile(conv_out)    # (batch, embed_dim)
        attn_out = self.sa(attn_in)      # (batch, embed_dim)
        return self.estimator(attn_out)


__all__ = [
    "ConvFilter",
    "SelfAttention",
    "EstimatorQNN",
    "HybridEstimatorQNN",
]
