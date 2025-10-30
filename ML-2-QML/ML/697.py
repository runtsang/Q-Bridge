"""Classical self‑attention module with multi-head support and dropout.

This module defines a SelfAttention class that implements a multi‑head
self‑attention mechanism using PyTorch. The class accepts an embedding
dimension, number of heads, and dropout probability. The ``run`` method
mirrors the signature of the original seed (rotation_params, entangle_params,
inputs) but ignores the quantum‑specific parameters; they are only used to
seed the weight initialization for reproducibility.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SelfAttention:
    """Multi‑head self‑attention with configurable dropout.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.q_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def _init_weights(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> None:
        """Seed the linear layers using the provided parameter arrays."""
        seed_q = int(np.sum(rotation_params) * 1e6) % (2**32)
        seed_k = int(np.sum(entangle_params) * 1e6) % (2**32)
        torch.manual_seed(seed_q)
        torch.nn.init.xavier_uniform_(self.q_linear.weight)
        torch.manual_seed(seed_k)
        torch.nn.init.xavier_uniform_(self.k_linear.weight)
        torch.nn.init.xavier_uniform_(self.v_linear.weight)
        torch.nn.init.xavier_uniform_(self.out_linear.weight)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array used only for seeding the weight initialization.
        entangle_params : np.ndarray
            Array used only for seeding the weight initialization.
        inputs : np.ndarray
            Input tensor of shape (seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            The output tensor of shape (seq_len, embed_dim).
        """
        self._init_weights(rotation_params, entangle_params)

        x = torch.as_tensor(inputs, dtype=torch.float32)
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        seq_len = x.shape[0]
        Q = Q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(0, 1).contiguous().view(seq_len, self.embed_dim)
        out = self.out_linear(out)
        return out.numpy()
