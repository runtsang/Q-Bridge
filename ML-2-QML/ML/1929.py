"""Enhanced multi‑head self‑attention with dropout and autograd support.

The class accepts a flattened weight vector for the Q, K, V projections and
a binary mask applied to the attention scores.  The interface mimics the
original seed (`run(rotation_params, entangle_params, inputs)`), but the
model now supports end‑to‑end training with PyTorch tensors.

Typical usage::

    sa = SelfAttentionEnhanced(embed_dim=64, num_heads=8, dropout=0.1)
    output = sa.run(rotation_params, entangle_params, inputs)

The module is self‑contained and can be dropped into any torch‑based
pipeline.

"""

__all__ = ["SelfAttentionEnhanced"]

import numpy as np
import torch
import torch.nn.functional as F


class SelfAttentionEnhanced:
    """Multi‑head self‑attention block with dropout on the attention
    scores.  `rotation_params` must contain
    3×num_heads×embed_dim×head_dim values that are reshaped into Q, K,
    and V projection matrices.  `entangle_params` is a binary mask of the
    same shape as the attention matrix and is applied element‑wise to the
    raw scores before the soft‑max.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        device: str | torch.device = "cpu",
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.device = torch.as_tensor([], dtype=torch.float32, device=device).device

        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")

    def _reshape_params(self, params: np.ndarray, shape: tuple) -> torch.Tensor:
        """Convert a flat numpy array to a torch tensor of given shape."""
        return torch.as_tensor(params.reshape(shape), dtype=torch.float32,
                               device=self.device)

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
        rotation_params : array, shape (3 * num_heads * embed_dim * head_dim,)
            Flattened weights for Q, K and V linear projections.
        entangle_params : array, shape (seq_len, seq_len)
            Binary mask applied element‑wise to the raw attention
            scores before the soft‑max.
        inputs : array, shape (batch, seq_len, embed_dim)
            Input embeddings.

        Returns
        -------
        numpy.ndarray
            Output of the attention block, shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape

        # Convert to tensors
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)

        # Reshape rotation_params into Q, K, V matrices
        # Each matrix shape: (num_heads, embed_dim, head_dim)
        per_head = self.num_heads * self.embed_dim * self.head_dim
        q_params = self._reshape_params(
            rotation_params[:per_head], (self.num_heads, self.embed_dim, self.head_dim)
        )
        k_params = self._reshape_params(
            rotation_params[per_head : 2 * per_head],
            (self.num_heads, self.embed_dim, self.head_dim),
        )
        v_params = self._reshape_params(
            rotation_params[2 * per_head :],
            (self.num_heads, self.embed_dim, self.head_dim),
        )

        # Linear projections: Q, K, V of shape (batch, num_heads, seq_len, head_dim)
        Q = torch.einsum("bse,heh->bhse", inputs_t, q_params)
        K = torch.einsum("bse,heh->bhse", inputs_t, k_params)
        V = torch.einsum("bse,heh->bhse", inputs_t, v_params)

        # Scaled dot‑product attention scores
        scores = torch.einsum("bhse,bhte->bhts", Q, K) / np.sqrt(self.head_dim)

        # Apply binary mask (broadcasting over batch)
        mask = torch.as_tensor(entangle_params, dtype=torch.float32,
                               device=self.device)
        scores = scores * mask

        # Soft‑max over the last dimension (sequence length)
        attn = F.softmax(scores, dim=-1)

        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        # Weighted sum of values
        context = torch.einsum("bhts,bhse->bhte", attn, V)

        # Concatenate heads and project back to embed_dim
        context = context.reshape(batch, seq_len, self.embed_dim)

        return context.detach().cpu().numpy()
