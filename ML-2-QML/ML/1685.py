"""Enhanced classical multi‑head self‑attention with dropout and residuals.

The implementation is intentionally lightweight so that the module can be used in a
short‑lived experimental pipeline or directly test‑based.  The class name is
`SelfAttention__gen263` to match the shared identifier used by the quantum
counterpart.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttention__gen263:
    """
    Classical multi‑head self‑attention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    n_heads : int, default 4
        Number of attention heads.
    dropout_prob : float, default 0.1
        Dropout probability applied to the attention output.
    layer_norm : bool, default False
        Whether to apply a final LayerNorm.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        dropout_prob: float = 0.1,
        layer_norm: bool = False,
    ):
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(embed_dim) if layer_norm else None

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
            Weight matrices for Q, K, V of shape ``(n_heads, 3, embed_dim, embed_dim)``.
            Each head has three matrices: Q, K, V.
        entangle_params : np.ndarray
            Unused in the classical implementation but kept for API compatibility.
        inputs : np.ndarray
            Input tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns
        -------
        np.ndarray
            The attended output of shape ``(batch, seq_len, embed_dim)``.
        """
        # Convert to torch tensors
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        rot = torch.as_tensor(rotation_params, dtype=torch.float32)

        batch, seq_len, _ = inp.shape
        # Split heads
        outputs = []
        for h in range(self.n_heads):
            Q = rot[h, 0]  # (..., embed_dim, embed_dim)
            K = rot[h, 1]
            V = rot[h, 2]
            # Compute Q, K, V
            q = inp @ Q
            k = inp @ K
            v = inp @ V
            # Scaled dot‑product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            outputs.append(out)
        # Concatenate heads
        concat = torch.cat(outputs, dim=-1)
        # Dropout
        concat = self.dropout(concat)
        # Residual connection
        out = concat + inp
        # Optional layer‑norm
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out.detach().cpu().numpy()

__all__ = ["SelfAttention__gen263"]
