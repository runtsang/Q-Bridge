"""Self‑attention module implemented purely with PyTorch.

Features:
* Variable embedding dimension and number of heads.
* Dropout applied to attention weights.
* Linear projection matrices derived from user‑supplied rotation
  and entangle parameters.
* Returns both the attention output and the weight matrix for
  inspection.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SelfAttentionModule:
    def __init__(
        self,
        embed_dim: int = 4,
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        """
        Parameters
        ----------
        embed_dim: int
            Total dimensionality of the input embeddings.
        num_heads: int
            Number of attention heads.  ``embed_dim`` must be divisible
            by ``num_heads``.
        dropout: float
            Dropout probability applied to the softmax attention
            weights.
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to (batch, heads, seq_len, head_dim)."""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of _split_heads."""
        batch, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch, seq_len, heads * head_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Execute multi‑head self‑attention.

        Parameters
        ----------
        rotation_params: np.ndarray
            Shape (embed_dim, embed_dim) used as the query projection matrix.
        entangle_params: np.ndarray
            Shape (embed_dim, embed_dim) used as the key projection matrix.
        inputs: np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        output: np.ndarray
            Attention‑weighted representation of the inputs.
        """
        # Convert to torch tensors
        q_w = torch.from_numpy(rotation_params).float()
        k_w = torch.from_numpy(entangle_params).float()
        v = torch.from_numpy(inputs).float()

        # Linear projections
        Q = v @ q_w.T
        K = v @ k_w.T
        V = v

        # Split into heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Scaled dot‑product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        context = torch.matmul(attn_weights, V)
        context = self._combine_heads(context)

        return context.detach().numpy()

__all__ = ["SelfAttentionModule"]
