"""Enhanced classical self‑attention with multi‑head support and dropout.

The API matches the original seed: a `SelfAttention()` factory returns an
instance exposing a `run(rotation_params, entangle_params, inputs)`
method.  The implementation uses PyTorch and can run on GPU if available.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSelfAttention:
    """Multi‑head self‑attention with optional linear pre‑processing."""
    def __init__(self, embed_dim: int = 64, num_heads: int = 8, dropout: float = 0.1):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Forward pass compatible with the original interface."""
        x = torch.from_numpy(inputs).float()
        # Optional linear pre‑processing
        if rotation_params is not None:
            rot = torch.from_numpy(rotation_params.reshape(self.embed_dim, self.embed_dim)).float()
            x = torch.matmul(x, rot)
        if entangle_params is not None:
            ent = torch.from_numpy(entangle_params.reshape(self.embed_dim, self.embed_dim)).float()
            x = torch.matmul(x, ent)

        qkv = self.qkv_proj(x)  # (batch, seq, 3*embed_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape for multi‑head
        q = q.view(-1, q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, q.size(1), self.embed_dim)

        out = self.out_proj(attn_output)
        return out.detach().cpu().numpy()

def SelfAttention():
    """Factory returning a ready‑to‑use instance."""
    return ClassicalSelfAttention(embed_dim=64, num_heads=8, dropout=0.1)

__all__ = ["SelfAttention"]
