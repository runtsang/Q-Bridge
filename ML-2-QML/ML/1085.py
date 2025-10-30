"""Enhanced multi-head self‑attention module built on PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention with optional dropout and layer‑norm.
    Mirrors the interface of the original seed but adds depth.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, rotation_params: torch.Tensor, entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor
            Dummy tensor to keep API compatible with the quantum version.
        entangle_params : torch.Tensor
            Dummy tensor to keep API compatible with the quantum version.

        Returns
        -------
        torch.Tensor
            Output of shape (batch, seq_len, embed_dim).
        """
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)  # (B, L, 3*embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for multi‑head
        q = q.contiguous().view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = k.contiguous().view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, L, L)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # (B, H, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        out = self.out_proj(attn_output)
        out = self.norm(out + x)  # residual + layer norm
        return out

__all__ = ["SelfAttention"]
