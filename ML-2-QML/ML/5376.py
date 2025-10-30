"""Hybrid self‑attention: classical core with optional quantum‑derived parameters.

The class below implements a multi‑head self‑attention block that accepts either
fully‑classical weight tensors or a set of rotation/entanglement parameters
produced by the quantum circuit.  The quantum parameters are reshaped into
linear projection matrices and used directly in the attention calculation,
enabling a clean interface for hybrid training pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Optional, Tuple

class ClassicalSelfAttention(nn.Module):
    """
    Multi‑head self‑attention with optional quantum‑derived projections.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads.
    bias : bool, optional
        Whether to add a learnable bias to the projections.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, bias: bool = True) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Classical weight matrices (optional)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: Optional[np.ndarray] = None,
        entangle_params: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Compute self‑attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray, optional
            1‑D array of length embed_dim * 3 * num_heads used to overwrite
            the classical query, key and value projections.
        entangle_params : np.ndarray, optional
            1‑D array of length num_heads * (seq_len - 1) used to modulate
            pairwise attention scores.

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape (batch, seq_len, embed_dim).
        """
        B, L, _ = inputs.shape

        # If quantum parameters are provided, replace the classical projection
        if rotation_params is not None:
            # Reshape into weight matrices for Q, K, V
            total_w = self.embed_dim * 3
            assert rotation_params.size == total_w, "rotation_params length mismatch"
            wq = torch.as_tensor(rotation_params[:self.embed_dim * self.embed_dim], dtype=torch.float32).reshape(
                self.embed_dim, self.embed_dim
            )
            wk = torch.as_tensor(rotation_params[self.embed_dim * self.embed_dim : 2 * self.embed_dim * self.embed_dim],
                                 dtype=torch.float32).reshape(self.embed_dim, self.embed_dim)
            wv = torch.as_tensor(rotation_params[2 * self.embed_dim * self.embed_dim :],
                                 dtype=torch.float32).reshape(self.embed_dim, self.embed_dim)
            q = inputs @ wq.T
            k = inputs @ wk.T
            v = inputs @ wv.T
        else:
            q = self.q_proj(inputs)
            k = self.k_proj(inputs)
            v = self.v_proj(inputs)

        # Reshape for multi‑head
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (B, H, L, L)

        # Optional entanglement modulation
        if entangle_params is not None:
            # reshape into a matrix that can be added to the scores
            assert entangle_params.size == self.num_heads * (L - 1), "entangle_params size mismatch"
            ent = torch.as_tensor(entangle_params, dtype=torch.float32).reshape(
                self.num_heads, L - 1
            )
            # broadcast addition along batch dimension
            attn_scores += ent.unsqueeze(0)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, L, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(attn_output)

__all__ = ["ClassicalSelfAttention"]
