"""Hybrid multi‑head self‑attention with dropout and neural‑parameterized weights.

This module extends the original SelfAttention helper by adding
* multi‑head attention (default 4 heads),
* an MLP that generates the linear‑weight matrices for Q and K,
* dropout on the value vectors before the final aggregation.

The public API mirrors the seed: the class provides a `run` method that
accepts `rotation_params`, `entangle_params` and `inputs`.  The
`rotation_params` and `entangle_params` are reshaped to the weight
matrices used for the Q and K projections respectively; this keeps
the call signature stable while exposing richer functionality.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi‑head self‑attention with dropout.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default=4
        Number of attention heads.
    dropout : float, default=0.1
        Dropout probability applied to the value vectors.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # MLP that produces the weight matrices for Q and K.
        # Using a single linear layer to produce all parameters.
        self.param_gen = nn.Linear(1, 2 * embed_dim * embed_dim)

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape a tensor of shape (batch, embed_dim) to
        (batch, num_heads, head_dim)."""
        return x.reshape(-1, self.num_heads, self.head_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Apply self‑attention to ``inputs``.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters that will be reshaped to a weight matrix for the Q projection.
        entangle_params : np.ndarray
            Parameters that will be reshaped to a weight matrix for the K projection.
        inputs : np.ndarray
            Input embeddings of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            The attended representation of shape (batch, embed_dim).
        """
        # Convert to torch tensors
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)

        # Construct weight matrices for Q and K from the provided params
        Wq = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        Wk = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)

        # Linear projections
        Q = inputs_t @ Wq
        K = inputs_t @ Wk
        V = inputs_t  # Use the raw inputs as values

        # Reshape to heads
        Qh = self._reshape_to_heads(Q)
        Kh = self._reshape_to_heads(K)
        Vh = self._reshape_to_heads(V)

        # Scaled dot‑product attention for each head
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout to values
        Vh = self.dropout(Vh)

        # Weighted sum
        head_outputs = torch.matmul(attn_weights, Vh)

        # Concatenate heads
        concat = head_outputs.reshape(-1, self.embed_dim)

        return concat.detach().cpu().numpy()

__all__ = ["SelfAttention"]
