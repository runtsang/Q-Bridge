"""Enhanced classical self‑attention with multi‑head, dropout, and residual connections."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

class SelfAttentionModule(nn.Module):
    """
    Multi‑head self‑attention module mirroring the original interface.
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
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projection layers
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Execute multi‑head self‑attention.
        Parameters
        ----------
        rotation_params : np.ndarray
            Unused in this implementation but retained for API compatibility.
        entangle_params : np.ndarray
            Unused in this implementation but retained for API compatibility.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).
        Returns
        -------
        np.ndarray
            Output tensor of the same shape as ``inputs``.
        """
        # Convert to torch tensor
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape for multi‑head attention
        B, N, _ = q.shape
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

        # Output projection and residual connection
        out = self.out_proj(out)
        out = self.layer_norm(out + x)
        return out.detach().numpy()

__all__ = ["SelfAttentionModule"]
