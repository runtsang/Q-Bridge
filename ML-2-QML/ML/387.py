"""Enhanced classical self‑attention with multi‑head support and dropout."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionUnit(nn.Module):
    """A multi‑head self‑attention block with optional dropout."""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch, seq_len, embed_dim).
        mask : torch.Tensor, optional
            Attention mask of shape (batch, seq_len, seq_len).

        Returns
        -------
        torch.Tensor
            Output sequence of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each shape (batch, seq_len, num_heads, head_dim)

        attn_scores = torch.einsum("bshd,bsHd->bshH", q, k) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum("bshH,bsHd->bshd", attn_weights, v)
        attn_output = attn_output.reshape(batch, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

__all__ = ["SelfAttentionUnit"]
