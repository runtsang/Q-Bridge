"""Hybrid self‑attention module integrating classical RBF kernel and an optional quantum kernel.

The class SelfAttentionHybrid implements a multi‑head attention mechanism where the attention
scores are derived from an RBF kernel over the query and key projections.  An optional
quantum kernel can be injected to augment or replace the classical similarity measure,
allowing seamless experimentation between purely classical and hybrid quantum/classical
attention flows.

Usage:
    attn = SelfAttentionHybrid(embed_dim=64, heads=8, gamma=0.1, use_quantum=False)
    out = attn(x)   # x shape: (batch, seq_len, embed_dim)
"""

import numpy as np
import torch
from torch import nn
from typing import Optional

class SelfAttentionHybrid(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int = 1,
        gamma: float = 1.0,
        use_quantum: bool = False,
        quantum_module: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.gamma = gamma
        self.use_quantum = use_quantum
        self.quantum_module = quantum_module

        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _rbf_kernel(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute a batched RBF kernel matrix between queries and keys."""
        # q, k: (batch, seq_len, d)
        diff = q.unsqueeze(2) - k.unsqueeze(1)  # (batch, seq_len, seq_len, d)
        dist_sq = torch.sum(diff * diff, dim=-1)  # (batch, seq_len, seq_len)
        return torch.exp(-self.gamma * dist_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor after applying hybrid attention.
        """
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        # Classical RBF attention scores
        scores = self._rbf_kernel(q, k)  # (batch, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)

        if self.use_quantum and self.quantum_module is not None:
            # Compute quantum kernel matrix; the quantum_module must expose a ``run`` method
            # that accepts two (seq_len, embed_dim) numpy arrays and returns a similarity matrix.
            quantum_scores = self.quantum_module.run(
                q.detach().cpu().numpy(),
                k.detach().cpu().numpy(),
            )
            quantum_scores = torch.tensor(quantum_scores, dtype=torch.float32, device=x.device)
            # Blend classical and quantum scores (simple average)
            attn_weights = 0.5 * attn_weights + 0.5 * torch.softmax(quantum_scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        return self.out_proj(out)

__all__ = ["SelfAttentionHybrid"]
