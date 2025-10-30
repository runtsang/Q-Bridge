"""Enhanced classical self‑attention module.

The implementation extends the original single‑head attention with
mult‑head support, dropout and a lightweight parameter‑tuning
interface.  Rotation and entangle parameters are interpreted as
weight matrices used to build the Q, K and V projections,
allowing a seamless transition to a hybrid workflow.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention with configurable dropout.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input sequence embeddings.
    num_heads : int, optional
        Number of attention heads.  Default is 4.
    dropout_prob : float, optional
        Dropout probability applied to the attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass for a single attention block.

        Parameters
        ----------
        rotation_params : np.ndarray
            Weight matrix used for Q, K and V projection.
            Shape (embed_dim, embed_dim).
        entangle_params : np.ndarray
            Optional bias added to the attention scores.
            If provided, shape must be (num_heads,).
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output of the attention block with the same shape as ``inputs``.
        """
        # Convert to torch tensors
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.q_proj.weight.device)

        # Apply projection with supplied rotation params
        rot_t = torch.as_tensor(rotation_params, dtype=torch.float32, device=x.device)
        ent_t = torch.as_tensor(entangle_params, dtype=torch.float32, device=x.device)
        q = torch.matmul(x, rot_t.T)
        k = torch.matmul(x, ent_t.T)
        v = x

        # Reshape for multi‑head
        batch, seq_len, _ = q.shape
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Optional bias from entangle_params
        if entangle_params.shape[0] == self.num_heads:
            bias = torch.as_tensor(entangle_params.reshape(self.num_heads, 1, 1), dtype=torch.float32)
            scores = scores + bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)

        return out.detach().cpu().numpy()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, dropout={self.dropout.p})"
        )


__all__ = ["SelfAttention"]
