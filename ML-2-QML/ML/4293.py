"""Hybrid self‑attention: classical branch.

This module defines a UnifiedSelfAttention class that implements a
classical self‑attention block.  The interface mirrors the quantum
counterpart so that a user can swap the two implementations.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

def SelfAttention():
    """Return a classical self‑attention module."""
    return UnifiedSelfAttention(embed_dim=4)

class UnifiedSelfAttention(nn.Module):
    """Classical self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.  The block expects
        tensors of shape (batch, seq_len, embed_dim).

    Attributes
    ----------
    query_proj : nn.Linear
    key_proj : nn.Linear
    value_proj : nn.Linear
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute self‑attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``inputs``.
        """
        Q = self.query_proj(inputs)
        K = self.key_proj(inputs)
        V = self.value_proj(inputs)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Legacy interface that accepts the same arguments as the quantum
        implementation.  ``rotation_params`` and ``entangle_params`` are
        ignored in the classical branch.

        Parameters
        ----------
        rotation_params : np.ndarray
            Unused.
        entangle_params : np.ndarray
            Unused.
        inputs : np.ndarray
            Input data of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output of the self‑attention block.
        """
        device = torch.device("cpu")
        with torch.no_grad():
            x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
            out = self.forward(x)
        return out.cpu().numpy()

__all__ = ["UnifiedSelfAttention", "SelfAttention"]
