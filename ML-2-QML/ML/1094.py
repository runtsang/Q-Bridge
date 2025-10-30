"""Enhanced classical self‑attention with multi‑head and dropout.

The class mirrors the original interface but now supports
multi‑head attention, optional masking, and dropout.
It also accepts a flattened array of rotation parameters that
are reshaped into the weight matrices of the linear layers.
"""

from __future__ import annotations

import math
import numpy as np
import torch


def SelfAttention():
    class MultiHeadSelfAttention:
        """Multi‑head self‑attention with dropout and optional masking."""
        def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
            if embed_dim % num_heads!= 0:
                raise ValueError("embed_dim must be divisible by num_heads")
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.dropout = dropout

            # Linear projections for Q, K, V, and output
            self.W_q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_k = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_v = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_o = torch.nn.Linear(embed_dim, embed_dim, bias=False)

            self.drop = torch.nn.Dropout(dropout)

        def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            """Compute multi‑head attention."""
            q = self.W_q(inputs)  # (batch, seq, embed)
            k = self.W_k(inputs)
            v = self.W_v(inputs)

            # Reshape for heads: (batch, num_heads, seq, head_dim)
            q = q.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
            k = k.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
            v = v.view(-1, self.num_heads, self.head_dim).transpose(0, 1)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = self.drop(attn)

            out = torch.matmul(attn, v)  # (num_heads, batch, seq, head_dim)
            out = out.transpose(0, 1).contiguous().view(-1, self.embed_dim)
            return self.W_o(out)

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
        ) -> np.ndarray:
            """
            Execute attention using a flattened array of rotation parameters.
            The array is interpreted as the concatenated weight matrices for
            W_q, W_k, and W_v.  entangle_params is retained for API
            compatibility but unused.
            """
            expected = 3 * self.embed_dim * self.embed_dim
            if rotation_params.size!= expected:
                raise ValueError(f"Expected {expected} rotation parameters, got {rotation_params.size}")

            # Load weights into the linear layers
            self.W_q.weight.data = torch.from_numpy(
                rotation_params[:self.embed_dim * self.embed_dim].reshape(self.embed_dim, self.embed_dim)
            )
            self.W_k.weight.data = torch.from_numpy(
                rotation_params[self.embed_dim * self.embed_dim : 2 * self.embed_dim * self.embed_dim].reshape(
                    self.embed_dim, self.embed_dim
                )
            )
            self.W_v.weight.data = torch.from_numpy(
                rotation_params[2 * self.embed_dim * self.embed_dim :].reshape(
                    self.embed_dim, self.embed_dim
                )
            )

            inp_tensor = torch.from_numpy(inputs).float()
            out_tensor = self.forward(inp_tensor)
            return out_tensor.detach().numpy()

    # Default configuration matches the original 4‑dimensional example
    return MultiHeadSelfAttention(embed_dim=4, num_heads=2, dropout=0.1)


__all__ = ["SelfAttention"]
