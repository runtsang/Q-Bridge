"""Multi‑head self‑attention with dropout and layer normalisation.

The class can be used as a drop‑in replacement for the simple 4‑dimensional
attention helper in the original seed.  It adds residual connections,
layer normalisation and optional dropout, making it suitable for
larger‑scale transformer architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention module that mirrors the classical interface
    while adding dropout, layer normalisation and optional residual
    connections.  The class can be used as a drop‑in replacement for the
    simple 4‑dimensional attention helper in the original seed.

    Parameters
    ----------
    embed_dim : int
        Dimension of the input embeddings.
    num_heads : int
        Number of attention heads.  Must divide ``embed_dim``.
    dropout : float, default 0.1
        Drop‑out probability applied to the attention weights.
    use_residual : bool, default True
        If ``True`` the input is added to the output (residual connection).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.residual = use_residual
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor | None = None,
        entangle_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(batch, seq_len, embed_dim)``.
        rotation_params, entangle_params : ignored
            Present only for API compatibility; they are **not** used
            in the classical implementation.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len, embed_dim)``.
        """
        batch, seq_len, _ = inputs.shape

        # Project into Q, K, V
        q = self.q_proj(inputs).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(inputs).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(inputs).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (batch, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        out = self.out_proj(out)
        if self.residual:
            out = out + inputs
        out = self.norm(out)
        return out

__all__ = ["SelfAttention"]
