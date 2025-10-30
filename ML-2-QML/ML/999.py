"""Enhanced classical multi‑head self‑attention with optional residual and layer‑norm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionEnhanced:
    """
    Multi‑head self‑attention module that mirrors the original API but adds
    dropout, layer‑norm and residual connections.  The constructor accepts
    ``embed_dim`` (total hidden size) and ``num_heads``.  ``rotation_params``
    and ``entangle_params`` are interpreted as weight matrices for the
    query/key projections and are reshaped accordingly.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape to (batch, seq_len, num_heads, head_dim)."""
        return tensor.view(tensor.size(0), tensor.size(1), self.num_heads, self.head_dim)

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
            Weight matrix for query projection. Shape (embed_dim, embed_dim).
        entangle_params : np.ndarray
            Weight matrix for key projection. Shape (embed_dim, embed_dim).
        inputs : np.ndarray
            Input sequence of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output of the attention block, shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape
        device = torch.device("cpu")

        # Convert to torch tensors
        x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        w_q = torch.as_tensor(rotation_params.reshape(self.embed_dim, self.embed_dim),
                              dtype=torch.float32, device=device)
        w_k = torch.as_tensor(entangle_params.reshape(self.embed_dim, self.embed_dim),
                              dtype=torch.float32, device=device)

        # Linear projections
        q = self._reshape(x @ w_q)
        k = self._reshape(x @ w_k)
        v = self._reshape(x)

        # Scaled dot‑product attention per head
        scores = torch.einsum("bshd,bsHd->bshH", q, k) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.einsum("bshH,bsHd->bshd", attn, v)
        context = context.reshape(batch, seq_len, self.embed_dim)

        # Residual + LayerNorm
        out = self.layer_norm(context + x)
        return out.cpu().numpy()

__all__ = ["SelfAttentionEnhanced"]
