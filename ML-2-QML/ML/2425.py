"""Hybrid classical convolution + self‑attention module.

This module extends the original Conv.py by adding a self‑attention
layer that operates on the convolution output.  The interface stays
compatible with the original Conv() factory: ``ConvAttention()`` returns
an object with a ``run`` method that accepts a 2‑D array and yields a
single scalar after convolution and attention aggregation.

The implementation uses a single‑channel 2‑D convolution followed by a
simple dot‑product self‑attention with a single attention head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ConvAttentionHybrid(nn.Module):
    """Convolution followed by self‑attention."""

    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        embed_dim: int = 4,
        attention_heads: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        # Linear projections for attention
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, data: np.ndarray) -> float:
        """Run convolution + attention on a 2‑D array."""
        # Convert to tensor and add batch/channel dims
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Convolution
        logits = self.conv(tensor)
        logits = torch.sigmoid(logits - self.conv_threshold)
        # Flatten spatial dimensions
        flat = logits.view(logits.size(0), -1)
        # Project to embedding space
        seq_len = flat.size(1)
        flat = flat.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        # Attention
        q = self.query_proj(flat)
        k = self.key_proj(flat)
        v = self.value_proj(flat)
        scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ v
        # Aggregate to scalar
        return attn_out.mean().item()

    def run(self, data: np.ndarray) -> float:
        """Compatibility wrapper for the original API."""
        return self.forward(data)


def ConvAttention():
    """Factory returning a ConvAttentionHybrid instance."""
    return ConvAttentionHybrid()


__all__ = ["ConvAttention", "ConvAttentionHybrid"]
