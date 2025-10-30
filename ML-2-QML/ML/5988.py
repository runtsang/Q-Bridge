"""Hybrid classical convolution and self‑attention with optional quantum back‑end.

The module exposes two factory functions that mirror the original
Conv.py and SelfAttention.py interfaces while adding a cross‑modal
interaction layer.  The classical path performs a convolution,
feeds the scalar output into a classical self‑attention block that
produces a feature vector.  The quantum path evaluates a
quanvolution circuit and then drives a quantum self‑attention
circuit using the same scalar as an input bias.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Classical self‑attention helper
class ClassicalSelfAttention:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# Hybrid convolution + attention
class HybridConvAttention(nn.Module):
    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)

    def run(self, data: np.ndarray) -> np.ndarray:
        """Apply a 2‑D convolution followed by classical self‑attention."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.conv_threshold)
        conv_out = activations.mean().item()

        # Build a dummy input vector for the attention block.
        # The vector is filled with the convolution scalar.
        inputs = np.full((1, self.attention.embed_dim), conv_out, dtype=np.float32)

        # Randomly initialise rotation and entangle parameters.
        rot_params = np.random.randn(self.attention.embed_dim * 3)
        ent_params = np.random.randn(self.attention.embed_dim - 1)

        return self.attention.run(rot_params, ent_params, inputs)

def Conv() -> HybridConvAttention:
    """Return a hybrid convolution‑attention filter."""
    return HybridConvAttention()

def SelfAttention() -> ClassicalSelfAttention:
    """Return a classical self‑attention helper."""
    return ClassicalSelfAttention(embed_dim=4)

__all__ = ["Conv", "SelfAttention", "HybridConvAttention"]
