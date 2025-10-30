"""Enhanced multi‑head self‑attention implementation.

Provides a PyTorch module that mirrors the legacy interface while
supporting residual connections, layer‑normalisation, dropout and
multiple attention heads.  The class is fully compatible with the
original ``SelfAttention`` helper, enabling drop‑in replacement in
existing pipelines."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention with residual, layer‑norm and dropout.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Drop‑out probability applied to the attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
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
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.size()
        qkv = self.qkv(x)          # (batch, seq_len, 3*embed_dim)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (batch, seq_len, num_heads, head_dim)

        # transpose to (batch, num_heads, seq_len, head_dim)
        q = q.permute(0, 3, 1, 4)
        k = k.permute(0, 3, 1, 4)
        v = v.permute(0, 3, 1, 4)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)  # (batch, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().reshape(batch, seq_len, self.embed_dim)
        out = self.out_proj(context)
        out = self.norm(out + x)  # residual + layer‑norm
        return out

__all__ = ["SelfAttention"]
