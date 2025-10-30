"""UnifiedHybridRegressor – classical‑plus‑attention regressor.

This module keeps EstimatorQNN’s lightweight API: a callable returning a
`UnifiedHybridRegressor` instance.  The class contains
* a vanilla 3‑layer MLP (as in the original EstimatorQNN),
* a self‑attention block that transforms the hidden representation,
* a linear head that produces the scalar output.

The architecture is intentionally simple enough for teaching but still
provides an example of how to fuse classical, quantum, and attention
components in one model.  It is fully compatible with the original
anchor and can be used in place of the EstimatorQNN class in the
existing code base.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class _SelfAttention(nn.Module):
    """Classical self‑attention inspired by SelfAttention.py."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, embed_dim)
        query = self.W_q(x)
        key   = self.W_k(x)
        value = self.W_v(x)
        scores = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5), dim=-1)
        return torch.matmul(scores, value)

class UnifiedHybridRegressor(nn.Module):
    """Hybrid regressor with a classical backbone, attention, and linear head."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, embed_dim: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, embed_dim),
        )
        self.attention = _SelfAttention(embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        attn_out = self.attention(hidden)
        return self.head(attn_out)

def EstimatorQNN() -> UnifiedHybridRegressor:
    return UnifiedHybridRegressor()

__all__ = ["UnifiedHybridRegressor", "EstimatorQNN"]
