import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionModule(nn.Module):
    """
    Classical multi‑head self‑attention module with optional dropout and
    layer‑norm.  The interface mirrors the original SelfAttention helper
    but provides a full transformer block suitable for downstream tasks.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input of shape (batch, seq_len, embed_dim).
        mask : Tensor, optional
            Boolean mask of shape (batch, seq_len) where True indicates
            positions to be masked (e.g., padding).  Masked positions are
            set to a large negative value before softmax.
        return_attention : bool, optional
            If True, also return attention weights.
        """
        batch, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*embed_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # reshape for multi‑head: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b, h, s, s)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, s)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (b, h, s, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        out = self.out_proj(attn_output)
        out = self.norm(out + x)  # residual + layer norm

        if return_attention:
            return out, attn_weights.mean(dim=1)  # average over heads
        return out

__all__ = ["SelfAttentionModule"]
