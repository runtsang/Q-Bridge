"""Enhanced classical self‑attention module.

This class extends the original seed by providing a multi‑head
attention implementation with dropout, batch support, and an explicit
``run`` method that accepts the same rotation and entanglement
matrices.  The module is fully torch‑based but exposes a NumPy
interface for easy integration with the quantum counterpart.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    """
    Multi‑head self‑attention with externally supplied weight matrices.

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
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Learnable linear projections (optional; not used in ``run``)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute attention output from batched inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray
            Weight matrix for queries, shape (embed_dim, embed_dim).
        entangle_params : np.ndarray
            Weight matrix for keys, shape (embed_dim, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention output of shape (batch, seq_len, embed_dim).
        """
        # Convert external matrices to torch tensors
        W_q_ext = torch.as_tensor(rotation_params, dtype=torch.float32)
        W_k_ext = torch.as_tensor(entangle_params, dtype=torch.float32)

        # Linear projections
        Q = inputs @ W_q_ext
        K = inputs @ W_k_ext
        V = inputs  # use raw values as in the seed

        # Reshape for multi‑head
        batch, seq_len, _ = Q.shape
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        # Final linear projection
        output = self.out_proj(context)
        return output

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Public API compatible with the original seed.

        Parameters
        ----------
        rotation_params : np.ndarray
            Matrix of shape (embed_dim, embed_dim).
        entangle_params : np.ndarray
            Matrix of shape (embed_dim, embed_dim).
        inputs : np.ndarray
            Input embeddings, shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Attention output as a NumPy array.
        """
        self.eval()
        with torch.no_grad():
            tensor_in = torch.as_tensor(inputs, dtype=torch.float32)
            out = self.forward(tensor_in, rotation_params, entangle_params)
        return out.numpy()


__all__ = ["SelfAttentionModule"]
