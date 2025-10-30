"""Classical multi‑head self‑attention module for transformer architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    """
    Multi‑head self‑attention with residual connection and layer norm.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the token embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to attention weights and output.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, value
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim) with residual
            connection and layer normalization applied.
        """
        batch, seq_len, _ = x.size()

        # [batch, seq_len, 3, heads, head_dim]
        qkv = self.qkv_proj(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        context = torch.matmul(attn, v)  # [batch, heads, seq_len, head_dim]
        context = context.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)

        out = self.out_proj(context)
        out = self.dropout(out)

        # Residual + LayerNorm
        return self.norm(out + x)
