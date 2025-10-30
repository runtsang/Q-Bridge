"""Hybrid multi‑head self‑attention with optional feed‑forward.

This module extends the original SelfAttention seed by adding:
- multi‑head attention (default 4 heads)
- optional feed‑forward network
- support for batched inputs and masking

The public API mirrors the seed (`run(rotation_params, entangle_params, inputs)`).
"""

import numpy as np
import torch
from torch import nn
from typing import Optional

class SelfAttention:
    """
    Classical multi‑head self‑attention with optional feed‑forward.
    """

    def __init__(self, embed_dim: int = 4, n_heads: int = 4, ffn_hidden: Optional[int] = None):
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        if embed_dim % n_heads!= 0:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.ffn_hidden = ffn_hidden

        # Optional feed‑forward network
        if ffn_hidden is not None:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_hidden),
                nn.ReLU(),
                nn.Linear(ffn_hidden, embed_dim),
            )
        else:
            self.ffn = None

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Execute the attention block.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_heads * embed_dim, ) – linear projection matrix for queries.
        entangle_params : np.ndarray
            Shape (n_heads * embed_dim, ) – linear projection matrix for keys.
        inputs : np.ndarray
            Shape (batch, seq_len, embed_dim).
        mask : np.ndarray, optional
            Boolean mask of shape (batch, seq_len). True indicates
            positions to mask out (set attention score to -inf).

        Returns
        -------
        np.ndarray
            Output of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape
        device = torch.device("cpu")

        # Convert to torch tensors
        x = torch.as_tensor(inputs, dtype=torch.float32, device=device)

        # Reshape parameter vectors into projection matrices
        rot = torch.as_tensor(rotation_params, dtype=torch.float32, device=device)
        proj = torch.as_tensor(entangle_params, dtype=torch.float32, device=device)

        rot = rot.reshape(self.n_heads, self.embed_dim, self.embed_dim)
        proj = proj.reshape(self.n_heads, self.embed_dim, self.embed_dim)

        # Compute per‑head projections
        query_list = []
        key_list = []
        value_list = []
        for h in range(self.n_heads):
            q = torch.matmul(x, rot[h])
            k = torch.matmul(x, proj[h])
            v = x
            query_list.append(q)
            key_list.append(k)
            value_list.append(v)

        query = torch.stack(query_list, dim=1)  # (batch, n_heads, seq_len, embed_dim)
        key = torch.stack(key_list, dim=1)
        value = torch.stack(value_list, dim=1)

        # Scaled dot‑product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
            mask = mask.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq_len, 1)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)

        context = torch.matmul(attn, value)  # (batch, n_heads, seq_len, embed_dim)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        # Optional feed‑forward
        if self.ffn is not None:
            context = self.ffn(context)

        return context.detach().numpy()

__all__ = ["SelfAttention"]
