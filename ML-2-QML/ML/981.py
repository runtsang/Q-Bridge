"""Robust multi‑head self‑attention module.

The class inherits from torch.nn.Module and exposes a `run` method that
mirrors the original interface but adds:

* multi‑head support (default 4 heads)
* optional dropout and masking
* ability to inject external rotation/entangle weight matrices
* automatic conversion between NumPy inputs and torch tensors
* gradient‑ready implementation for end‑to‑end training

This design turns the simple helper into a drop‑in replacement for
standard Transformer attention layers while keeping the original
signature for compatibility with downstream code.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Multi‑head scaled dot‑product self‑attention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Drop‑out probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "embed_dim must be divisible by num_heads"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: torch.Tensor | None = None,
        entangle_params: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the attention output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor, optional
            External weight matrix for the query projection.
            Shape must be (embed_dim, embed_dim).
        entangle_params : torch.Tensor, optional
            External weight matrix for the key projection.
            Shape must be (embed_dim, embed_dim).
        mask : torch.Tensor, optional
            Boolean mask of shape (batch, seq_len, seq_len).

        Returns
        -------
        out : torch.Tensor
            Output of shape (batch, seq_len, embed_dim).
        attn_weights : torch.Tensor
            Attention weights of shape (batch, num_heads, seq_len, seq_len).
        """
        if rotation_params is not None:
            Q = x @ rotation_params.reshape(self.embed_dim, self.embed_dim)
        else:
            Q = self.q_linear(x)

        if entangle_params is not None:
            K = x @ entangle_params.reshape(self.embed_dim, self.embed_dim)
        else:
            K = self.k_linear(x)

        V = self.v_linear(x)

        # reshape for multi‑head
        batch, seq_len, _ = Q.shape
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot‑product
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_linear(attn_output)

        return out, attn_weights

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compatibility wrapper that accepts NumPy arrays.

        Parameters
        ----------
        rotation_params : np.ndarray
            External rotation weight matrix.
        entangle_params : np.ndarray
            External entangle weight matrix.
        inputs : np.ndarray
            Input array of shape (batch, seq_len, embed_dim).
        mask : np.ndarray, optional
            Boolean mask array.

        Returns
        -------
        np.ndarray
            Attention output as NumPy array.
        """
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(inputs, dtype=torch.float32)
            rot = torch.as_tensor(rotation_params, dtype=torch.float32)
            ent = torch.as_tensor(entangle_params, dtype=torch.float32)
            msk = torch.as_tensor(mask, dtype=torch.bool) if mask is not None else None
            out, _ = self.forward(x, rot, ent, msk)
            return out.cpu().numpy()

__all__ = ["SelfAttention"]
