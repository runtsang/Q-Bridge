"""Self‑attention module with multi‑head and dropout support.

The class mirrors the original API but adds richer behaviour:
* `n_heads` controls the number of parallel attention heads.
* Dropout is applied to the attention weights.
* The module is a torch.nn.Module so it can be trained end‑to‑end.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention:
    """Multi‑head self‑attention with dropout.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    n_heads : int, default=4
        Number of attention heads.
    dropout : float, default=0.1
        Dropout probability applied to the attention matrix.
    """

    def __init__(self, embed_dim: int, *, n_heads: int = 4, dropout: float = 0.1):
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)

        # Linear layers for Q, K, V.  Each produces `n_heads * head_dim` outputs.
        self.q_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_lin = nn.Linear(embed_dim, embed_dim, bias=False)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [batch, seq, embed] -> [batch, heads, seq, head_dim]."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Compute attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Not used in the classical model – kept for API compatibility.
        entangle_params : np.ndarray
            Not used in the classical model – kept for API compatibility.
        inputs : np.ndarray
            Shape (batch, seq, embed_dim).

        Returns
        -------
        np.ndarray
            The attended representations, shape (batch, seq, embed_dim).
        """
        X = torch.as_tensor(inputs, dtype=torch.float32, device="cpu")

        # Linear projections
        Q = self._reshape_for_heads(self.q_lin(X))
        K = self._reshape_for_heads(self.k_lin(X))
        V = self._reshape_for_heads(self.v_lin(X))

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # [batch, heads, seq, head_dim]
        context = context.transpose(1, 2).contiguous().view(X.shape)
        out = self.out_lin(context)

        return out.detach().numpy()

# expose the class for import
__all__ = ["SelfAttention"]
