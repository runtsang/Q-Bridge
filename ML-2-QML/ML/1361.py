"""Enhanced classical self‑attention module.

This implementation extends the original seed by adding multi‑head
attention, dropout, bias handling and an explicit interface that
mirrors the quantum version.  It remains fully compatible with
PyTorch‑based training pipelines and can be dropped into existing
transformer stacks without modification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ClassicalSelfAttention(nn.Module):
    """
    Classical self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 1
        Number of attention heads.  The implementation splits the
        embedding dimension evenly across heads; if ``embed_dim`` is
        not divisible by ``num_heads`` a ValueError is raised.
    dropout : float, default 0.0
        Dropout probability applied to the attention weights.
    bias : bool, default True
        Whether to add learnable bias terms to the linear projections.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1,
                 dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Storage for the last computed attention weights
        self._last_weights: Optional[torch.Tensor] = None

    def forward(self,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor,
                inputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the self‑attention output.

        Parameters
        ----------
        rotation_params : torch.Tensor
            Weight matrix for the query projection.  Shape
            ``(embed_dim, embed_dim)``.  This replaces the linear layer
            used in the seed; it allows the caller to supply a custom
            projection (e.g. from a quantum circuit).
        entangle_params : torch.Tensor
            Weight matrix for the key projection.  Shape
            ``(embed_dim, embed_dim)``.
        inputs : torch.Tensor
            Input tensor of shape ``(batch, seq_len, embed_dim)``.
        mask : torch.Tensor, optional
            Boolean mask of shape ``(batch, seq_len)`` where ``True``
            indicates positions to be ignored.  The mask is applied
            before the softmax.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, seq_len, embed_dim)``.
        """
        batch, seq_len, _ = inputs.shape

        # Custom linear projections
        q = F.linear(inputs, rotation_params)
        k = F.linear(inputs, entangle_params)
        v = inputs

        # Reshape for multi‑head attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        self._last_weights = attn_weights

        context = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        return context

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the attention weights from the most recent forward call."""
        return self._last_weights

def SelfAttention() -> ClassicalSelfAttention:
    """Factory mirroring the seed interface."""
    return ClassicalSelfAttention(embed_dim=4, num_heads=1, dropout=0.0)

__all__ = ["SelfAttention"]
