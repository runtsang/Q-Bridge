"""Enhanced multi‑head self‑attention module with dropout and PyTorch support.

The class implements the same public interface as the original seed but adds
support for multiple attention heads, optional dropout, and back‑propagation
through the attention block.  It can be dropped into a larger PyTorch model
without modification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModule(nn.Module):
    def __init__(self, embed_dim: int, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Input of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape

        # Linear projections
        q = self.q_proj(inputs).reshape(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(inputs).reshape(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(inputs).reshape(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # (batch, heads, seq_len, head_dim)
        context = context.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)

        return self.out_proj(context)

def SelfAttention(embed_dim: int, heads: int = 1, dropout: float = 0.0):
    """
    Factory returning a ready‑to‑use instance of SelfAttentionModule.
    """
    return SelfAttentionModule(embed_dim=embed_dim, heads=heads, dropout=dropout)

__all__ = ["SelfAttention", "SelfAttentionModule"]
