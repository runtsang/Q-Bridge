"""Enhanced multi‑head self‑attention module using PyTorch."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SelfAttention:
    """
    Multi‑head self‑attention with optional dropout.
    Parameters are supplied as weight matrices for the query and key projections.
    The value projection is the identity.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        device: torch.device | str | None = None,
    ):
        """
        Args:
            embed_dim: Dimensionality of the input embeddings.
            num_heads: Number of attention heads.
            dropout: Dropout probability applied to the attention weights.
            device: Torch device (defaults to CPU).
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.device = torch.device(device or "cpu")

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the multi‑head self‑attention output.

        Parameters
        ----------
        rotation_params: shape (num_heads, embed_dim, embed_dim)
            Weight matrices for the query projection per head.
        entangle_params: shape (num_heads, embed_dim, embed_dim)
            Weight matrices for the key projection per head.
        inputs: shape (seq_len, embed_dim)
            Input token embeddings.

        Returns
        -------
        output: shape (seq_len, embed_dim)
            Self‑attention weighted sum of the inputs.
        """
        seq_len = inputs.shape[0]
        # Convert to torch tensors
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        q_params = torch.as_tensor(rotation_params, dtype=torch.float32, device=self.device)
        k_params = torch.as_tensor(entangle_params, dtype=torch.float32, device=self.device)

        # Split heads: (num_heads, seq_len, head_dim)
        q = torch.einsum("hij,nsj->hns", q_params, x.unsqueeze(0))
        k = torch.einsum("hij,nsj->hns", k_params, x.unsqueeze(0))

        # Scale and compute scores
        scores = torch.einsum("hns,hms->hnm", q, k) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)

        if self.dropout > 0.0:
            scores = F.dropout(scores, p=self.dropout, training=True)

        # Value projection is identity; concatenate heads
        v = x.unsqueeze(0)  # (1, seq_len, embed_dim)
        v = v.repeat(self.num_heads, 1, 1)  # (num_heads, seq_len, embed_dim)

        out = torch.einsum("hnm,hms->hns", scores, v)  # (num_heads, seq_len, embed_dim)
        out = out.reshape(seq_len, self.embed_dim)  # (seq_len, embed_dim)

        return out.cpu().numpy()


__all__ = ["SelfAttention"]
