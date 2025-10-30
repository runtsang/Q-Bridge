import numpy as np
import torch
from torch import nn

class SelfAttention:
    """
    Multi‑head, trainable self‑attention module that mirrors the original
    interface: ``run(rotation_params, entangle_params, inputs)``.
    The angle arrays are accepted for API compatibility but ignored.
    """
    def __init__(self, embed_dim: int, num_heads: int = 2, depth: int = 1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        # Depth‑wise feed‑forward before projections
        self.ff = nn.Sequential(
            *[nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(depth)]
        )

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)"""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        rotation_params, entangle_params : np.ndarray
            Accepted for compatibility but ignored.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        attn_weights : np.ndarray
            Attention probability matrix of shape (batch, seq_len, seq_len).
        context : np.ndarray
            Weighted sum of value vectors, shape (batch, seq_len, embed_dim).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        x = self.ff(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        # Scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, v)
        # Merge heads
        context = context.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.embed_dim)

        return attn_weights.numpy(), context.numpy()
