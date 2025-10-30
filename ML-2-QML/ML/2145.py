"""Enhanced classical self‑attention with multi‑head and dropout support."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class ClassicalSelfAttention(nn.Module):
    """
    Multi‑head self‑attention with optional dropout.
    The interface mirrors the original seed: ``run`` accepts
    rotation and entangle parameters together with the input tensor.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.device = device or torch.device("cpu")
        self.to(self.device)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes attention output following the transformer formulation.
        ``rotation_params`` and ``entangle_params`` are unused in the
        classical computation but kept for API compatibility.
        """
        B, T, _ = inputs.shape
        Q = self.q_proj(inputs).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(inputs).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(inputs).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(attn_output)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        API compatible wrapper that accepts NumPy inputs and returns a NumPy array.
        """
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        rotation_t = torch.as_tensor(rotation_params, dtype=torch.float32, device=self.device)
        entangle_t = torch.as_tensor(entangle_params, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out_t = self.forward(inputs_t, rotation_t, entangle_t)
        return out_t.cpu().numpy()


def SelfAttention():
    """
    Factory that returns a pre‑configured ClassicalSelfAttention instance.
    """
    return ClassicalSelfAttention(embed_dim=4, num_heads=2, dropout=0.1)


__all__ = ["SelfAttention"]
