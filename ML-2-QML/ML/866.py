"""Classical multi‑head self‑attention module with residual connections and dropout.

This implementation generalises the original single‑head attention to multiple heads,
adds dropout for regularisation, and includes a residual skip connection to ease
gradient flow.  It accepts batched sequences and returns the attended representation
with the same shape as the input, making it drop‑in compatible with transformer
backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to the attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim) with a residual
            connection added.
        """
        batch, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*embed_dim)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (batch, heads, seq_len, head_dim)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale  # (batch, heads, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.einsum("bhqk,bhkd->bhqd", attn, v)  # (batch, heads, seq_len, head_dim)
        context = context.permute(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)
        out = self.out_proj(context)
        return out + x  # residual connection

__all__ = ["SelfAttention"]
