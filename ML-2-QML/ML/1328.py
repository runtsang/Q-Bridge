"""Enhanced classical self‑attention with multi‑head, dropout and layer‑norm."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention block with optional dropout and layer‑norm.
    The interface mirrors the original seed but adds richer functionality.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm

        # Linear projections for Q, K, V – parameters are packed into rotation_params
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        if use_layer_norm:
            self.ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Flattened weights for Q, K and V projections.
            Shape: (3 * embed_dim, )
        entangle_params : np.ndarray
            Unused in the classical version but kept for API compatibility.
        inputs : np.ndarray
            Input sequence of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output of shape (batch, seq_len, embed_dim).
        """
        # Load parameters into linear layers
        q_w = rotation_params[: self.embed_dim * self.embed_dim].reshape(
            self.embed_dim, self.embed_dim
        )
        k_w = rotation_params[
            self.embed_dim * self.embed_dim : 2 * self.embed_dim * self.embed_dim
        ].reshape(self.embed_dim, self.embed_dim)
        v_w = rotation_params[
            2 * self.embed_dim * self.embed_dim : 3 * self.embed_dim * self.embed_dim
        ].reshape(self.embed_dim, self.embed_dim)

        self.q_proj.weight.data = torch.tensor(q_w, dtype=torch.float32)
        self.k_proj.weight.data = torch.tensor(k_w, dtype=torch.float32)
        self.v_proj.weight.data = torch.tensor(v_w, dtype=torch.float32)

        # Forward pass
        batch, seq_len, _ = inputs.shape
        x = torch.as_tensor(inputs, dtype=torch.float32)

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)

        if self.use_layer_norm:
            out = self.ln(out)

        return out.detach().numpy()
