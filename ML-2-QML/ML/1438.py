"""Classical self‑attention with multi‑head support and optional dropout.

The class mirrors the quantum interface: ``run(rotation_params, entangle_params, inputs)``.
"""

import numpy as np
import torch
import torch.nn.functional as F

class SelfAttentionBlock:
    """
    Multi‑head self‑attention with configurable dropout.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        num_heads : int, optional
            Number of attention heads. Defaults to 1.
        dropout : float, optional
            Dropout probability applied to the attention scores. Defaults to 0.0.
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compute multi‑head attention.

        Parameters
        ----------
        rotation_params : array-like
            Linear projection parameters for Q, K, V.
            Shape: (3 * embed_dim,).
        entangle_params : array-like
            Unused in the classical version but kept for API compatibility.
        inputs : array-like
            Input sequence of shape (batch, seq_len, embed_dim).

        Returns
        -------
        output : ndarray
            Attention output of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape
        # Linear projections
        proj = torch.as_tensor(inputs, dtype=torch.float32)
        proj = proj @ torch.as_tensor(rotation_params.reshape(3, self.embed_dim, -1), dtype=torch.float32)
        q, k, v = proj[:, :, :self.embed_dim], proj[:, :, self.embed_dim:2*self.embed_dim], proj[:, :, 2*self.embed_dim:]
        # Reshape for heads
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        if self.dropout > 0.0:
            scores = F.dropout(scores, p=self.dropout, training=True)
        out = torch.matmul(scores, v)
        # Merge heads
        out = out.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        return out.numpy()
