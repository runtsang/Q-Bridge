"""Extended classical self‑attention with multi‑head support and learnable projection matrices.

The implementation stays fully classical, using PyTorch for efficient tensor operations.
It mirrors the interface of the original seed while adding:
  • configurable number of attention heads
  • explicit projection matrices for Q, K, V
  • batch‑friendly execution
"""

import numpy as np
import torch
import torch.nn.functional as F

class SelfAttention:
    """Multi‑head self‑attention block."""
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Projection parameters: Q, K, V combined into a single vector
        self._rotation_params = np.random.randn(3 * self.embed_dim).astype(np.float32)
        self._entangle_params = np.random.randn(self.embed_dim).astype(np.float32)

    @property
    def rotation_params(self):
        return self._rotation_params

    @property
    def entangle_params(self):
        return self._entangle_params

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Flattened projection parameters for Q, K and V.
            Expected shape : (3 * embed_dim)
        entangle_params : np.ndarray
            Optional bias parameters per head (unused but kept for API compatibility).
            Expected shape : (embed_dim)
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output of shape (batch, seq_len, embed_dim) after attention fusion.
        """
        batch, seq_len, _ = inputs.shape
        q_proj = rotation_params.reshape(self.embed_dim, -1)
        k_proj = rotation_params.reshape(self.embed_dim, -1)
        v_proj = rotation_params.reshape(self.embed_dim, -1)

        q = torch.matmul(inputs, q_proj)
        k = torch.matmul(inputs, k_proj)
        v = torch.matmul(inputs, v_proj)

        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=False)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        return out.detach().cpu().numpy()

__all__ = ["SelfAttention"]
