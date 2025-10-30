"""Classical self‑attention with trainable projections and multi‑head support."""
from __future__ import annotations

import torch
import torch.nn as nn

class SelfAttentionModule(nn.Module):
    """
    A multi‑head self‑attention layer suitable for inclusion in transformer backbones.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int, default 4
        Number of attention heads. Must divide ``embed_dim``.
    dropout : float, default 0.1
        Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(batch_size, seq_len, embed_dim)``

        Returns
        -------
        torch.Tensor
            Same shape as ``inputs``.
        """
        B, T, _ = inputs.shape

        # Project to Q, K, V
        q = self.q_proj(inputs).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(inputs).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(inputs).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, v)  # (B, heads, T, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, self.embed_dim)

        # Final projection
        return self.out_proj(attn_output)

    def run(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compatibility wrapper mimicking the original interface.

        Parameters
        ----------
        rotation_params : torch.Tensor
            Ignored in the classical version but accepted for API parity.
        entangle_params : torch.Tensor
            Ignored in the classical version but accepted for API parity.
        inputs : torch.Tensor
            Input embeddings.

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation.
        """
        # In the classical implementation the params are not used
        return self.forward(inputs)

__all__ = ["SelfAttentionModule"]
