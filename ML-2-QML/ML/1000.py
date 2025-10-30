"""Enhanced self‑attention module with multi‑head support and optional dropout.

The class mirrors the original interface but adds richer functionality:
* multi‑head attention with configurable number of heads
* dropout on the attention weights
* convenience ``run`` method that accepts rotation and entangle parameters
  (used by the quantum counterpart) and maps them to linear projections.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi‑head self‑attention with optional dropout.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default=1
        Number of attention heads.
    dropout : float, default=0.0
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard multi‑head self‑attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(context)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Convenience wrapper that maps the supplied rotation and entangle
        parameters to linear projections and runs the attention forward pass.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the Q projection. Shape ``(embed_dim * embed_dim,)``.
        entangle_params : np.ndarray
            Parameters for the K projection. Shape ``(embed_dim * embed_dim,)``.
        inputs : np.ndarray
            Input tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns
        -------
        np.ndarray
            Output of the attention block as a NumPy array.
        """
        # Reshape parameters into weight matrices
        q_weight = rotation_params.reshape(self.embed_dim, self.embed_dim)
        k_weight = entangle_params.reshape(self.embed_dim, self.embed_dim)

        # Override the learned projections with the supplied matrices
        self.q_proj.weight.data = torch.from_numpy(q_weight.T).float()
        self.k_proj.weight.data = torch.from_numpy(k_weight.T).float()

        # Forward pass using the overridden weights
        x = torch.from_numpy(inputs).float()
        with torch.no_grad():
            out = self.forward(x)
        return out.numpy()

__all__ = ["SelfAttention"]
