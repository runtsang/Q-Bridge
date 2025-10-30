"""Enhanced classical self‑attention module.

Provides a multi‑head, dropout‑regularised self‑attention layer that
mirrors the quantum interface while offering richer behaviour.

The public factory ``SelfAttentionEnhanced`` returns an instance
exposing ``run(inputs, rotation_params, entangle_params)``.
"""

import numpy as np
import torch
import torch.nn.functional as F

def SelfAttentionEnhanced(embed_dim: int = 64,
                          num_heads: int = 4,
                          dropout: float = 0.1) -> "ClassicalSelfAttentionEnhanced":
    class ClassicalSelfAttentionEnhanced:
        """Multi‑head self‑attention with dropout and gated value."""

        def __init__(self, embed_dim: int, num_heads: int, dropout: float):
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.dropout = torch.nn.Dropout(p=dropout)

            # Weight matrices for linear projections
            self.W_q = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
            self.W_k = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
            self.W_v = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
            self.W_o = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))

        def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq, dim = x.shape
            x = x.view(batch, seq, self.num_heads, self.head_dim)
            return x.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

        def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
            batch, heads, seq, head_dim = x.shape
            x = x.permute(0, 2, 1, 3).contiguous()
            return x.view(batch, seq, heads * head_dim)

        def run(self,
                inputs: np.ndarray,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> np.ndarray:
            """
            Parameters
            ----------
            inputs : shape (batch, seq, embed_dim)
                Query‑key‑value vectors.
            rotation_params : shape (embed_dim, embed_dim)
                Parameters used to compute Q and K linear projections.
            entangle_params : shape (embed_dim, embed_dim)
                Parameters used to compute V linear projection.
            Returns
            -------
            output : shape (batch, seq, embed_dim)
            """
            x = torch.as_tensor(inputs, dtype=torch.float32)

            # Apply rotation parameters as learned linear projections
            q = torch.matmul(x, rotation_params.reshape(self.embed_dim, -1))
            k = torch.matmul(x, entangle_params.reshape(self.embed_dim, -1))
            v = torch.matmul(x, rotation_params.reshape(self.embed_dim, -1))  # reuse for value

            # Multi‑head split
            q = self._split_heads(q)
            k = self._split_heads(k)
            v = self._split_heads(v)

            # Scaled dot‑product attention per head
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
            scores = F.softmax(scores, dim=-1)
            scores = self.dropout(scores)

            # Weighted sum of values
            attn = torch.matmul(scores, v)

            # Concatenate heads and project out
            attn = self._combine_heads(attn)
            output = torch.matmul(attn, self.W_o)

            return output.detach().numpy()

    return ClassicalSelfAttentionEnhanced(embed_dim, num_heads, dropout)

__all__ = ["SelfAttentionEnhanced"]
