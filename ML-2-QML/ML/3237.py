"""Hybrid classical self‑attention with convolutional feature extraction.

This module implements a drop‑in replacement for the original SelfAttention
class, but augments it with a learnable 2‑D convolutional filter.  The
convolutional front‑end is inspired by the Conv.py seed and is followed by
the classical self‑attention block from SelfAttention.py.  The interface
mirrors the original so that existing code can import ``SelfAttentionModel``
and call ``run`` without modification.

The module is intentionally lightweight: it only requires PyTorch and
NumPy, and it can be used with or without GPU acceleration.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Convolutional front‑end (adapted from Conv.py)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Learned 2‑D convolutional filter that reduces the input to a scalar."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Shape ``(batch, embed_dim)``.  Each embedding is interpreted
            as a 2‑D image of size ``kernel_size`` for the convolution.

        Returns
        -------
        torch.Tensor
            Scalar activation per batch element.
        """
        # Reshape to a 2‑D image and apply convolution
        data = data.view(data.size(0), 1, self.kernel_size, self.kernel_size)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])  # scalar per batch

# --------------------------------------------------------------------------- #
# Classical self‑attention core (adapted from SelfAttention.py)
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Linear self‑attention block that operates on the concatenated
    embedding and convolutional features."""
    def __init__(self, embed_dim: int, kernel_size: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = ConvFilter(kernel_size=kernel_size)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(batch, embed_dim)``.
        rotation_params, entangle_params : torch.Tensor
            Parameters for the linear projections.  They are interpreted as
            weight matrices of shape ``(embed_dim, embed_dim)``.
        """
        # Convolutional feature extraction
        conv_out = self.conv(inputs)
        # Broadcast back to the embedding dimension
        conv_extended = conv_out.unsqueeze(-1).expand(-1, self.embed_dim)
        # Combine with raw embeddings
        combined = inputs + conv_extended

        # Linear projections
        query = self.query_proj(combined)
        key   = self.key_proj(combined)
        value = self.value_proj(combined)

        # Scaled dot‑product attention
        scores = torch.softmax((query @ key.t()) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

# --------------------------------------------------------------------------- #
# Public API wrapper
# --------------------------------------------------------------------------- #
class SelfAttentionModel:
    """Classical self‑attention model that can be dropped into existing pipelines.

    The constructor accepts the embedding dimension and the convolutional
    kernel size.  The ``run`` method mirrors the original interface and
    returns a NumPy array.
    """
    def __init__(self, embed_dim: int = 4, kernel_size: int = 2) -> None:
        self.model = ClassicalSelfAttention(embed_dim=embed_dim, kernel_size=kernel_size)

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        inputs : np.ndarray
            Shape ``(batch, embed_dim)``.
        rotation_params, entangle_params : np.ndarray
            Weight matrices flattened to 1‑D arrays of length
            ``embed_dim * embed_dim``.  They are reshaped inside the class.
        """
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        rot = torch.as_tensor(rotation_params.reshape(self.model.embed_dim, -1), dtype=torch.float32)
        ent = torch.as_tensor(entangle_params.reshape(self.model.embed_dim, -1), dtype=torch.float32)
        out = self.model(inp, rot, ent)
        return out.detach().numpy()

__all__ = ["SelfAttentionModel"]
