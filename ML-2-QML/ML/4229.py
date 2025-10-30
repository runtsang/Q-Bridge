"""Hybrid ConvGen220: classical convolution + self‑attention.

This module defines a drop‑in replacement for Conv.py that
- applies a trainable 2‑D convolution,
- embeds the flattened feature map into a lower‑dimensional space,
- runs a classical self‑attention mechanism, and
- produces a two‑class probability vector.

The implementation is intentionally lightweight so that it can be swapped
for a quantum version without altering the surrounding pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalSelfAttention:
    """Simple self‑attention used by ConvGen220."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class ConvGen220(nn.Module):
    """Hybrid convolutional feature extractor with attention."""
    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        attention_dim: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Linear projection to the attention dimension
        self.embed = None  # created in forward based on input shape
        self.attention_dim = attention_dim

        # Learnable attention parameters
        self.rotation_params = nn.Parameter(
            torch.randn(attention_dim, attention_dim), requires_grad=True
        )
        self.entangle_params = nn.Parameter(
            torch.randn(attention_dim, attention_dim), requires_grad=True
        )
        self.attention = ClassicalSelfAttention(attention_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, H, W)
        Returns:
            Tensor of shape (B, 2) containing class probabilities.
        """
        # Step 1: Convolution + sigmoid threshold
        conv_out = torch.sigmoid(self.conv(x) - self.conv_threshold)

        # Step 2: Flatten and embed
        B, C, H, W = conv_out.shape
        flat = conv_out.view(B, -1)
        if self.embed is None or self.embed.in_features!= flat.size(1):
            self.embed = nn.Linear(flat.size(1), self.attention_dim, bias=True)
            self.add_module("embed", self.embed)
        embedded = self.embed(flat)

        # Step 3: Classical self‑attention
        attn_output = self.attention.run(
            self.rotation_params.detach().cpu().numpy(),
            self.entangle_params.detach().cpu().numpy(),
            embedded.detach().cpu().numpy(),
        )
        attn_tensor = torch.as_tensor(attn_output, dtype=torch.float32, device=x.device)

        # Step 4: Final probability
        prob = torch.sigmoid(attn_tensor.mean(dim=1, keepdim=True))
        return torch.cat((prob, 1 - prob), dim=1)


__all__ = ["ConvGen220"]
