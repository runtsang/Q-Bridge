import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionGen013(nn.Module):
    """Trainable multi‑head self‑attention module.

    Parameters
    ----------
    embed_dim : int
        Dimension of the input embeddings.
    num_heads : int, optional
        Number of attention heads. Must divide embed_dim. Default: 4.
    dropout : float, optional
        Drop‑out probability applied to attention weights. Default: 0.1.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for queries, keys, values
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output sequence of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.shape

        # Shape: (batch, seq_len, 3, num_heads, head_dim)
        qkv = self.qkv_proj(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (batch, seq_len, num_heads, head_dim)

        # Scaled dot‑product attention
        scores = torch.einsum("bshd,bshd->bhs", q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        attn_out = torch.einsum("bhs,bshd->bshd", attn, v)
        attn_out = attn_out.reshape(batch, seq_len, self.embed_dim)

        # Final linear projection
        return self.out_proj(attn_out)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """Compatibility wrapper mirroring the original seed interface."""
        return self.forward(x)

__all__ = ["SelfAttentionGen013"]
