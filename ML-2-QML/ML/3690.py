"""Hybrid classical self‑attention with quantum‑inspired estimator.

The class implements a multi‑head self‑attention block followed by a lightweight
regression head. It merges the SelfAttention seed (classical attention logic)
and the EstimatorQNN seed (simple feed‑forward regression) into a single
torch.nn.Module."""

import numpy as np
import torch
from torch import nn

class HybridSelfAttentionEstimator(nn.Module):
    """
    Multi‑head self‑attention followed by a regression head.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads.
    hidden_dim : int
        Hidden size of the regression network.
    """
    def __init__(self, embed_dim: int = 4, num_heads: int = 2, hidden_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Regression head inspired by EstimatorQNN
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, embed_dim)

        Returns
        -------
        Tensor of shape (batch, 1) – regression output.
        """
        # Compute Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch, seq_len, _ = q.shape

        # Reshape for multi‑head attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (batch, heads, seq_len, seq_len)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)  # (batch, heads, seq_len, head_dim)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        # Pool over sequence and project
        out = self.out_proj(context).mean(dim=1)  # (batch, embed_dim)

        # Regression head
        return self.regressor(out)

__all__ = ["HybridSelfAttentionEstimator"]
