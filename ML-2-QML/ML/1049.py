"""Multi‑head self‑attention with optional dropout and parameter‑tunable
rotations and entanglements.  The interface mimics the original seed but
adds trainable linear projections and a dropout layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SelfAttentionModule(nn.Module):
    """
    Classical multi‑head self‑attention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Optional linear layers to incorporate rotation/entangle params
        self.rotation_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.entangle_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim) and transpose.
        """
        batch, seq_len, embed_dim = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge the head dimension back into the embedding dimension.
        """
        batch, num_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, num_heads * head_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: Optional[torch.Tensor] = None,
        entangle_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the self‑attention output.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor, optional
            Additional parameters to be added to the query projection.
        entangle_params : torch.Tensor, optional
            Additional parameters to be added to the key projection.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        # Linear projections
        Q = self.q_proj(inputs)
        K = self.k_proj(inputs)
        V = self.v_proj(inputs)

        # Incorporate optional params
        if rotation_params is not None:
            Q = Q + self.rotation_proj(rotation_params)
        if entangle_params is not None:
            K = K + self.entangle_proj(entangle_params)

        # Split heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = self._merge_heads(attn_output)

        # Final linear projection
        return self.out_proj(attn_output)

    def run(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compatibility wrapper matching the original seed interface.
        """
        return self.forward(inputs, rotation_params, entangle_params)


def SelfAttention() -> SelfAttentionModule:
    """
    Factory function returning a pre‑configured attention module.
    """
    return SelfAttentionModule(embed_dim=4, num_heads=2, dropout=0.1)


__all__ = ["SelfAttentionModule", "SelfAttention"]
