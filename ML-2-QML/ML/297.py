"""Enhanced classical self‑attention with multi‑head support and residual connections.

The class mirrors the original interface but adds:
* configurable number of heads
* dropout regularisation
* residual addition and layer normalisation
* optional linear projection of the rotation and entangle parameters
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionExtended:
    """
    Multi‑head self‑attention block that accepts the same signature as the seed.
    Parameters:
        embed_dim (int): Dimensionality of the token embeddings.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to (batch, heads, seq_len, head_dim)."""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass of the multi‑head attention.
        rotation_params and entangle_params are interpreted as linear projection weights
        and are ignored if they do not match the expected shape.
        """
        # Convert inputs to tensor
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Optional linear mapping of the provided params
        if rotation_params.size in {self.embed_dim, self.embed_dim * 3}:
            q_weight = rotation_params.reshape(self.embed_dim, self.embed_dim)
            self.q_proj.weight.data = torch.from_numpy(q_weight)
        if entangle_params.size in {self.embed_dim, self.embed_dim * 3}:
            k_weight = entangle_params.reshape(self.embed_dim, self.embed_dim)
            self.k_proj.weight.data = torch.from_numpy(k_weight)

        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        # Scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.shape)

        # Output projection, residual and norm
        out = self.out_proj(attn_output)
        out = self.norm(x + out)

        return out.detach().numpy()


__all__ = ["SelfAttentionExtended"]
