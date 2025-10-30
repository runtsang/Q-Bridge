"""Unified self‑attention module mirroring the original SelfAttention helper but with multi‑head, dropout, and a placeholder for quantum projection.

This class is entirely classical and uses PyTorch for efficient tensor operations.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UnifiedSelfAttention"]

class UnifiedSelfAttention(nn.Module):
    """
    Classical multi‑head self‑attention that optionally reserves slots for quantum
    projection matrices.  The public ``run`` API mirrors the original helper,
    accepting rotation and entanglement parameters that are ignored in the
    classical implementation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_qc: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the token embeddings.
        num_heads : int
            Number of attention heads.
        dropout : float
            Drop‑out applied to the attention scores.
        use_qc : bool
            Flag reserved for future quantum integration; currently has no effect.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_qc = use_qc

        # classical linear projections
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def _project(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        """Project and reshape for multi‑head attention."""
        return linear(x).view(x.size(0), -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)

    def _compute_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Scaled dot‑product attention with dropout."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.num_heads)
        scores = self.dropout(scores)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi‑head self‑attention.
        """
        Q = self._project(inputs, self.q_linear)
        K = self._project(inputs, self.k_linear)
        V = self._project(inputs, self.v_linear)

        out = self._compute_attention(Q, K, V)
        return out.transpose(1, 2).contiguous().view(inputs.shape[0], -1, self.embed_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compatibility wrapper that accepts the same signature as the original
        SelfAttention helper.  The rotation and entanglement parameters are
        ignored in the classical implementation.
        """
        return self.forward(torch.as_tensor(inputs, dtype=torch.float32)).numpy()
