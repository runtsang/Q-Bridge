"""Enhanced classical self‑attention module with PyTorch support."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Classical self‑attention layer.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    dropout : float, optional
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor | np.ndarray,
        entangle_params: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """
        Compute self‑attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Input embeddings of shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor | np.ndarray
            Parameters for the query projection. Shape must be
            (embed_dim, embed_dim).
        entangle_params : torch.Tensor | np.ndarray
            Parameters for the key projection. Shape must be
            (embed_dim, embed_dim).

        Returns
        -------
        torch.Tensor
            Output of the attention mechanism with shape (batch, seq_len, embed_dim).
        """
        if isinstance(rotation_params, np.ndarray):
            rotation_params = torch.as_tensor(rotation_params, dtype=torch.float32)
        if isinstance(entangle_params, np.ndarray):
            entangle_params = torch.as_tensor(entangle_params, dtype=torch.float32)

        # Apply custom linear projections
        Q = F.linear(inputs, rotation_params.t())
        K = F.linear(inputs, entangle_params.t())
        V = self.v_proj(inputs)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        return torch.matmul(attn, V)

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Convenience wrapper that accepts NumPy arrays and returns a NumPy array.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(
                torch.as_tensor(inputs, dtype=torch.float32),
                torch.as_tensor(rotation_params, dtype=torch.float32),
                torch.as_tensor(entangle_params, dtype=torch.float32),
            )
        return out.cpu().numpy()


__all__ = ["SelfAttention"]
