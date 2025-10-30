"""Enhanced classical self‑attention with multi‑head support and optional parameter initialization.

This module defines a PyTorch `SelfAttention` class inheriting from `nn.Module`.  
It supports multiple heads, dropout, and can optionally take NumPy arrays
(`rotation_params`, `entangle_params`) to initialize the linear projections
in a single call.  The interface mimics the original seed while providing
a richer, trainable implementation suitable for integration into larger
neural architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention module.

    Args:
        embed_dim (int): Dimensionality of input embeddings.
        num_heads (int, optional): Number of attention heads. Defaults to 1.
        dropout (float, optional): Dropout probability on attention weights. Defaults to 0.0.
        use_param_init (bool, optional): If True and parameter arrays are supplied,
            initialise projection weights from them. Defaults to False.
        rotation_params (np.ndarray, optional): Shape (embed_dim, embed_dim)
            used to initialise query, value projections.
        entangle_params (np.ndarray, optional): Shape (embed_dim, embed_dim)
            used to initialise key, output projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        use_param_init: bool = False,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        if use_param_init and rotation_params is not None and entangle_params is not None:
            self._init_from_params(rotation_params, entangle_params)

    def _init_from_params(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Initialize projection weights from supplied NumPy arrays."""
        rot = rotation_params.reshape(self.embed_dim, self.embed_dim)
        ent = entangle_params.reshape(self.embed_dim, self.embed_dim)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.from_numpy(rot))
            self.k_proj.weight.copy_(torch.from_numpy(ent))
            self.v_proj.weight.copy_(torch.from_numpy(rot))
            self.out_proj.weight.copy_(torch.from_numpy(ent))

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Compute multi‑head self‑attention.

        Args:
            inputs (torch.Tensor): Shape (batch, seq_len, embed_dim).
            rotation_params (np.ndarray, optional): For on‑the‑fly weight init.
            entangle_params (np.ndarray, optional): For on‑the‑fly weight init.

        Returns:
            torch.Tensor: Shape (batch, seq_len, embed_dim).
        """
        if rotation_params is not None or entangle_params is not None:
            self._init_from_params(rotation_params, entangle_params)

        B, L, D = inputs.shape
        Q = self.q_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(inputs).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(context)


__all__ = ["SelfAttention"]
