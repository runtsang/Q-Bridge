"""Enhanced classical self‑attention with multi‑head support and learnable projections."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class SelfAttentionEnhanced(nn.Module):
    """A classical multi‑head self‑attention layer that mirrors the quantum interface.

    The layer accepts a dictionary of parameters similar to the quantum variant:
    ``params = {"inputs": inputs, "rotation_params": None, "entangle_params": None}``
    The rotation and entangle parameters are ignored but kept for API compatibility.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 1,
                 dropout: float = 0.0,
                 device: str | None = None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        if device:
            self.to(device)

    def forward(self, params: dict) -> np.ndarray:
        """Run the attention block.

        Parameters
        ----------
        params : dict
            Must contain the key ``"inputs"`` with shape ``(batch, seq_len, embed_dim)``.
            ``"rotation_params"`` and ``"entangle_params"`` are ignored but accepted
            for API compatibility.

        Returns
        -------
        np.ndarray
            The attended representation of shape ``(batch, seq_len, embed_dim)``.
        """
        inputs = params["inputs"]
        batch, seq_len, _ = inputs.shape

        q = self.q_proj(inputs).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(inputs).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(inputs).view(batch, seq_len, self.num_heads, self.head_dim)

        # transpose to (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return out.detach().cpu().numpy()

__all__ = ["SelfAttentionEnhanced"]
