"""Hybrid classical self‑attention with convolutional preprocessing.

The class exposes a `run` method that takes a 2‑D input array, a set of rotation
parameters, and entanglement parameters.  Internally a small Conv2d filter
extracts local features, which are then fed into a standard
scaled‑dot‑product attention mechanism.  The design mirrors the
`SelfAttention.py` seed while extending it with a ConvFilter inspired by
`Conv.py`.  The implementation remains fully NumPy/PyTorch compatible
for quick prototyping or integration into larger pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ConvFilter(nn.Module):
    """Simple 2×2 convolutional filter with learnable bias."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.squeeze()


class HybridSelfAttention:
    """Classical self‑attention with optional convolutional preprocessing."""
    def __init__(self, embed_dim: int, kernel_size: int = 2, conv_threshold: float = 0.0):
        self.embed_dim = embed_dim
        self.conv = ConvFilter(kernel_size=kernel_size, threshold=conv_threshold)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Flat array of length 3 * embed_dim, reshaped into (embed_dim, 3) for query/key.
        entangle_params : np.ndarray
            Flat array of length embed_dim - 1, used for entanglement scaling.
        inputs : np.ndarray
            2‑D array representing the feature map to attend over.

        Returns
        -------
        np.ndarray
            Attention‑weighted output of shape (embed_dim,).
        """
        # Convolutional preprocessing
        conv_out = self.conv.forward(inputs).numpy()

        # Build query, key, value matrices
        query = torch.as_tensor(
            conv_out @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            conv_out @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(conv_out, dtype=torch.float32)

        # Scaled dot‑product attention
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        weighted = scores @ value
        return weighted.numpy()


__all__ = ["HybridSelfAttention"]
