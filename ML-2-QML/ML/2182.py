"""SelfAttentionGen404 – a multi‑head, dropout‑aware self‑attention implementation for PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SelfAttentionGen404(nn.Module):
    """
    Multi‑head self‑attention with optional dropout and a linear output layer.
    The class is deliberately structured to mirror the quantum interface:
        * ``run`` accepts rotation_params, entangle_params, and inputs
          and returns the attention output as a NumPy array.
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
        self.dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Final linear projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: torch.Tensor | None = None,
                entangle_params: torch.Tensor | None = None) -> torch.Tensor:
        """
        Standard multi‑head attention forward pass.
        ``rotation_params`` and ``entangle_params`` are ignored in the classical
        implementation but kept for API compatibility with the quantum version.
        """
        B, N, _ = inputs.shape

        # Projecting into Q, K, V
        Q = self.q_proj(inputs).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = self.k_proj(inputs).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(inputs).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5  # (B, H, N, N)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # (B, H, N, D)
        context = context.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

        return self.out_proj(context)

    def run(self,
            rotation_params: torch.Tensor,
            entangle_params: torch.Tensor,
            inputs: torch.Tensor) -> np.ndarray:
        """
        Compatibility wrapper to match the quantum ``run`` signature.
        Returns a NumPy array, enabling downstream hybrid pipelines.
        """
        with torch.no_grad():
            out = self.forward(inputs, rotation_params, entangle_params)
        return out.cpu().numpy()


__all__ = ["SelfAttentionGen404"]
