"""Hybrid convolution‑attention module for classical deep learning.

The class combines a 2‑D convolutional filter (with a sigmoid activation
controlled by a threshold) and a self‑attention block.  It mirrors the
interface of the quantum implementation so that it can be swapped
without changing client code."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ClassicalSelfAttention(nn.Module):
    """Simple feed‑forward self‑attention implemented with PyTorch."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        q = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        k = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        v = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ v).detach().numpy()


class HybridConvAttention(nn.Module):
    """Drop‑in replacement for the quantum quanvolution + attention block."""
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        embed_dim: int = 4,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.attention = ClassicalSelfAttention(embed_dim)

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: 2‑D array of shape (kernel_size, kernel_size)

        Returns:
            Output of the attention block as a 1‑D NumPy array.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        conv_mean = activations.mean().item()

        # Construct dummy rotation/entangle parameters that reflect the conv output
        rot = np.full((self.attention.embed_dim,), conv_mean, dtype=np.float32)
        ent = np.full((self.attention.embed_dim,), conv_mean, dtype=np.float32)
        inp = np.ones((self.attention.embed_dim,), dtype=np.float32)

        return self.attention(rot, ent, inp)


__all__ = ["HybridConvAttention"]
