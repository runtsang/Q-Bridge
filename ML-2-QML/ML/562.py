"""Enhanced classical self‑attention module.

Provides a multi‑head, gated attention mechanism that interprets the
rotation and entangle parameters as linear projection weights, allowing a
direct comparison with the quantum implementation.
"""

import torch
import torch.nn as nn
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Multi‑head self‑attention with gating and optional dropout."""

    def __init__(self, embed_dim: int = 4, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers that will be seeded by the rotation_params
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Gating layer to modulate the attention output
        self.gate = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def run(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        rotation_params : torch.Tensor
            Tensor of shape (3 * embed_dim,) used to initialise ``qkv_proj`` weights.
        entangle_params : torch.Tensor
            Tensor of shape (embed_dim,) used to initialise ``out_proj`` weights.
        inputs : torch.Tensor
            Input embeddings of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output of the gated attention block.
        """
        # Initialise projection weights from supplied parameters
        with torch.no_grad():
            self.qkv_proj.weight.copy_(rotation_params.view(3 * self.embed_dim, self.embed_dim))
            self.out_proj.weight.copy_(entangle_params.view(self.embed_dim, self.embed_dim))

        # Compute Q, K, V
        qkv = self.qkv_proj(inputs)  # (batch, seq_len, 3*embed_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Scale queries
        q = q / np.sqrt(self.head_dim)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # (batch, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)  # (batch, seq_len, embed_dim)

        # Output projection
        out = self.out_proj(attn_output)

        # Gating
        gate = torch.sigmoid(self.gate(out))
        return gate * out

def SelfAttention():
    return ClassicalSelfAttention()

__all__ = ["SelfAttention"]
