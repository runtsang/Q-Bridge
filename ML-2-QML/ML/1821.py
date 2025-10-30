"""Enhanced classical self‑attention module with PyTorch backend.

Provides a flexible, trainable self‑attention layer exposing the same
interface as the original seed.  The implementation supports
embedding dimension, key/value/query projection matrices, dropout,
layer normalisation and optional masking.  The class can be used
directly in a torch.nn.Module or instantiated and run on NumPy
arrays for quick inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SelfAttentionGen422(nn.Module):
    """Layer‑wise self‑attention with trainable projections."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6, device=device)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rotation_params: Optional[torch.Tensor] = None,
        entangle_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, embed_dim)
            Input embeddings.
        mask : Tensor of shape (batch, seq_len, seq_len) or None
            Optional attention mask; True indicates positions to be ignored.
        rotation_params, entangle_params : ignored
            Placeholder to retain API compatibility with the original
            quantum‑style signature.  They are not used in the
            classical implementation but are accepted so that the same
            call can be used in a hybrid workflow.
        """
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        out = self.norm(out)
        return out

    def run(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility wrapper to match the seed API."""
        # rotation_params and entangle_params are unused; they are kept
        # to preserve the call signature used by the quantum counterpart.
        return self.forward(inputs)

__all__ = ["SelfAttentionGen422"]
