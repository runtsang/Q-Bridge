"""Multi‑head self‑attention module for PyTorch.

Provides a parameterised attention layer that can be dropped‑in
where a single‑head self‑attention was previously used.
"""

import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    """Multi‑head scaled dot‑product self‑attention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to attention scores.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, embed_dim).

        Returns
        -------
        out : torch.Tensor
            Output of shape (batch, seq_len, embed_dim).
        attn : torch.Tensor
            Attention weights of shape (batch, num_heads, seq_len, seq_len).
        """
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, batch, heads, seq_len, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return out, attn

def SelfAttention(embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
    """Factory that returns a multi‑head self‑attention module."""
    return MultiHeadSelfAttention(embed_dim, num_heads, dropout)

__all__ = ["SelfAttention"]
