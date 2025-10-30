"""Enhanced classical self‑attention supporting multi‑head and residual connections."""

import numpy as np
import torch


class SelfAttention:
    """
    Classical multi‑head self‑attention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, optional
        Number of attention heads. Must divide embed_dim.
    residual : bool, optional
        Add residual connection from input to output.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, residual: bool = False):
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.residual = residual
        self.head_dim = embed_dim // num_heads

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
        rotation_params : ndarray of shape (embed_dim, embed_dim)
            Parameters used to compute queries.
        entangle_params : ndarray of shape (embed_dim, embed_dim)
            Parameters used to compute keys.
        inputs : ndarray of shape (batch, embed_dim)
            Input embeddings.

        Returns
        -------
        output : ndarray of shape (batch, embed_dim)
            Attention output (optionally with residual).
        """
        batch, _ = inputs.shape
        # Reshape params for heads
        rot = rotation_params.reshape(self.num_heads, self.head_dim, self.head_dim)
        ent = entangle_params.reshape(self.num_heads, self.head_dim, self.head_dim)
        inp = inputs.reshape(batch, self.num_heads, self.head_dim)

        # Compute Q, K, V per head
        Q = torch.from_numpy(inp) @ torch.from_numpy(rot)
        K = torch.from_numpy(inp) @ torch.from_numpy(ent)
        V = torch.from_numpy(inp)

        # Attention scores
        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.head_dim), dim=-1)
        out = scores @ V  # shape (batch, num_heads, head_dim)

        # Concatenate heads
        out = out.reshape(batch, self.embed_dim)

        if self.residual:
            out = out + torch.from_numpy(inputs)

        return out.numpy()


__all__ = ["SelfAttention"]
