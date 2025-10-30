"""Enhanced multi‑head self‑attention with dropout and optional quantum‑derived attention weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionGen299(nn.Module):
    """
    Classical multi‑head self‑attention block.
    Parameters
    ----------
    embed_dim: int
        Dimensionality of input embeddings.
    num_heads: int, default 4
        Number of attention heads.
    dropout: float, default 0.1
        Dropout probability applied to attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor of shape (batch, seq_len, embed_dim)
            Input embeddings.
        Returns
        -------
        torch.Tensor of shape (batch, seq_len, embed_dim)
            Output after self‑attention.
        """
        B, N, _ = x.shape
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_linear(attn_out)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Legacy interface that mirrors the original seed.
        The rotation_params and entangle_params are interpreted as linear weight matrices
        for the query and key projections respectively.
        """
        with torch.no_grad():
            q_w = torch.tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            k_w = torch.tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            x = torch.tensor(inputs, dtype=torch.float32)
            q = x @ q_w.t()
            k = x @ k_w.t()
            v = x
            scores = F.softmax(q @ k.t() / (self.embed_dim ** 0.5), dim=-1)
            return (scores @ v).numpy()

def SelfAttention() -> SelfAttentionGen299:
    """
    Factory that returns a ready‑to‑use multi‑head attention instance.
    """
    return SelfAttentionGen299(embed_dim=4, num_heads=2, dropout=0.1)

__all__ = ["SelfAttentionGen299", "SelfAttention"]
