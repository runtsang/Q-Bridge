"""Hybrid classical‑quantum estimator combining feed‑forward regression and self‑attention.

This module extends the original EstimatorQNN by adding a classical self‑attention
block inspired by the SelfAttention seed.  The resulting network can be used
directly in a PyTorch training loop and exposes the attention parameters
for inspection or transfer to a quantum front‑end.
"""

import torch
from torch import nn
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention layer using matrix multiplication and softmax."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Parameters for linear projections
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weighted sum."""
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ V

class HybridEstimatorQNN(nn.Module):
    """Feed‑forward regressor with an optional self‑attention head."""
    def __init__(self, input_dim: int = 2, embed_dim: int = 4):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention then regression."""
        attn_out = self.attention(x)
        return self.regressor(attn_out)

def EstimatorQNN():
    """Compatibility wrapper returning the hybrid model."""
    return HybridEstimatorQNN()

__all__ = ["HybridEstimatorQNN", "EstimatorQNN"]
