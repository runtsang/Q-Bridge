"""Enhanced classical self‑attention block with trainable projections and optional masking.

The class implements a multi‑head attention mechanism that can be trained end‑to‑end with
PyTorch. It mirrors the interface of the original seed but adds learnable query,
key, and value projections, dropout, and an optional causal mask. The `run`
method accepts a NumPy array of inputs and returns a NumPy array of the attention
output, while the `forward` method is a standard `nn.Module` interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionEnhanced(nn.Module):
    """Trainable multi‑head self‑attention block."""

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to (batch, num_heads, seq_len, head_dim)."""
        return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)

    def _attention(self, q, k, v, mask=None):
        """Compute scaled dot‑product attention."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).
        mask : torch.Tensor, optional
            Binary mask of shape (batch, seq_len, seq_len) where 0 indicates
            positions to ignore.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).
        """
        q = self._reshape_for_heads(self.q_proj(inputs))
        k = self._reshape_for_heads(self.k_proj(inputs))
        v = self._reshape_for_heads(self.v_proj(inputs))

        attn = self._attention(q, k, v, mask)
        attn = attn.transpose(1, 2).contiguous().view(inputs.size(0), inputs.size(1), self.embed_dim)
        return self.out_proj(attn)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper that mimics the seed interface.

        Parameters
        ----------
        rotation_params : np.ndarray
            Unused in this classical implementation; accepted for API compatibility.
        entangle_params : np.ndarray
            Unused in this classical implementation; accepted for API compatibility.
        inputs : np.ndarray
            Input array of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Attention output of shape (batch, seq_len, embed_dim).
        """
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.forward(inp)
        return out.detach().cpu().numpy()

__all__ = ["SelfAttentionEnhanced"]
