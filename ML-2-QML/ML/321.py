"""Enhanced classical self‑attention with multi‑head, dropout, and optional positional encoding.

The class mirrors the original interface but adds richer capabilities:
* configurable number of heads
* dropout on the attention scores
* sinusoidal positional embeddings
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionGen357:
    """Multi‑head self‑attention with optional dropout and positional encoding."""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1,
                 use_pos_encoding: bool = True):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_pos_encoding = use_pos_encoding

        # Linear projections
        self.w_q = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.w_k = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.w_v = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.out_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # Positional encoding
        self.pos_enc = None
        if self.use_pos_encoding:
            self.pos_enc = self._sinusoidal_pos_encoding(max_len=2048, d_model=embed_dim)

    def _sinusoidal_pos_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compute multi‑head self‑attention.  `rotation_params` and `entangle_params`
        are accepted for API compatibility but are ignored in the classical
        implementation.

        Parameters
        ----------
        rotation_params : np.ndarray
            Unused in this implementation.
        entangle_params : np.ndarray
            Unused in this implementation.
        inputs : np.ndarray
            Shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Shape (batch, seq_len, embed_dim).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        if self.use_pos_encoding:
            seq_len = x.shape[1]
            x = x + self.pos_enc[:seq_len]

        # Linear projections
        q = torch.matmul(x, self.w_q)
        k = torch.matmul(x, self.w_k)
        v = torch.matmul(x, self.w_v)

        # Reshape for multi‑head
        q = q.view(x.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(x.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(x.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        scores = F.dropout(scores, p=self.dropout, training=True)

        context = torch.matmul(scores, v)
        context = context.transpose(1, 2).contiguous().view(x.shape[0], -1, self.embed_dim)
        output = torch.matmul(context, self.out_proj)
        return output.detach().cpu().numpy()

__all__ = ["SelfAttentionGen357"]
