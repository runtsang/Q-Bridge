"""Hybrid self‑attention module with learnable scaling and optional quantum similarity integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionModule(nn.Module):
    """
    A transformer‑style self‑attention block that extends the original design by:
        * exposing a learnable scaling matrix per head,
        * supporting multiple heads,
        * optionally fusing quantum‑derived similarity scores.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, use_quantum: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Learnable scaling per head
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Flag for quantum similarity integration
        self.use_quantum = use_quantum
        self.quantum_func = None

    def set_quantum_func(self, func):
        """Attach a callable that returns a (batch, seq_len, seq_len) similarity tensor."""
        self.quantum_func = func

    def forward(self, x: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, L, embed_dim)
            Input embeddings.
        rotation_params : Tensor
            Parameters for the quantum rotation gates.
        entangle_params : Tensor
            Parameters for the quantum entangling gates.

        Returns
        -------
        Tensor (B, L, embed_dim)
            Self‑attention output.
        """
        B, L, _ = x.size()

        # Linear projections
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Fuse quantum similarity if available
        if self.use_quantum and self.quantum_func is not None:
            quantum_scores = self.quantum_func(rotation_params, entangle_params)
            quantum_scores = quantum_scores.unsqueeze(1)  # (B, 1, L, L)
            attn_logits = attn_logits + quantum_scores

        # Apply learnable scaling
        attn_logits = attn_logits * self.scale

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)

        return attn_output

__all__ = ["SelfAttentionModule"]
