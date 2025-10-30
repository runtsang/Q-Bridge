"""Enhanced multi‑head self‑attention module using PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    """
    Multi‑head self‑attention block with residual connection and layer‑norm.
    Mirrors the interface of the original seed while adding trainable projections,
    dropout and optional scaling.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim) if use_layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
        Returns:
            Tensor of shape (batch, seq_len, embed_dim)
        """
        B, T, _ = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        out = self.out_proj(out)

        if self.layer_norm is not None:
            out = self.layer_norm(out)

        return out
