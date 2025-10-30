"""
Enhanced classical self‑attention module with multi‑head and dropout.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionEnhanced(nn.Module):
    """
    Multi‑head self‑attention with optional parameter sharing.
    The rotation_params and entangle_params can override the linear
    projections for query and key respectively, enabling a shared
    weight interface with the quantum implementation.
    """
    def __init__(self, embed_dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Final linear projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray = None,
                entangle_params: np.ndarray = None) -> torch.Tensor:
        """
        inputs: Tensor of shape (batch, seq_len, embed_dim)
        rotation_params: optional weight matrix for q_proj (flattened)
        entangle_params: optional weight matrix for k_proj (flattened)
        """
        batch_size, seq_len, _ = inputs.shape

        # Override q_proj weights if rotation_params provided
        if rotation_params is not None:
            weight_q = torch.as_tensor(rotation_params.reshape(self.embed_dim, self.embed_dim),
                                       dtype=inputs.dtype, device=inputs.device)
            self.q_proj.weight = nn.Parameter(weight_q)

        # Override k_proj weights if entangle_params provided
        if entangle_params is not None:
            weight_k = torch.as_tensor(entangle_params.reshape(self.embed_dim, self.embed_dim),
                                       dtype=inputs.dtype, device=inputs.device)
            self.k_proj.weight = nn.Parameter(weight_k)

        # Linear projections
        q = self.q_proj(inputs)  # (batch, seq_len, embed_dim)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)

        # Reshape for multi‑head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        context = torch.matmul(scores, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Final projection
        output = self.out_proj(context)
        return output


def SelfAttention():
    """
    Factory that returns an instance of SelfAttentionEnhanced with
    default hyper‑parameters.
    """
    return SelfAttentionEnhanced(embed_dim=4, num_heads=2, dropout=0.1)
