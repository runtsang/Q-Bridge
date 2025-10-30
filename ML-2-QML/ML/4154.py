"""Hybrid classical layer combining fully connected, self‑attention and regression.

The design merges the fully connected layer from FCL, the self‑attention mechanism from
SelfAttention, and the regression head of EstimatorQNN.  All parameters are trainable
torch.nn.Parameters and the module can be used inside a standard PyTorch training loop.
"""

import torch
from torch import nn
import numpy as np

class HybridLayer(nn.Module):
    """
    A hybrid neural block:
        * Linear projection (FC) from input_dim to hidden_dim
        * Self‑attention over the hidden representation
        * Small regression head producing a scalar output
    """
    def __init__(self, input_dim: int, hidden_dim: int = 8, attn_dim: int = 4):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        # Self‑attention parameters
        self.q_proj = nn.Linear(hidden_dim, attn_dim)
        self.k_proj = nn.Linear(hidden_dim, attn_dim)
        self.v_proj = nn.Linear(hidden_dim, attn_dim)
        # Regression head
        self.out = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc(x))
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1)), dim=-1)
        attn = torch.matmul(scores, v)
        return self.out(attn)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper matching the `run` interface of the seed examples."""
        return self.forward(x)

__all__ = ["HybridLayer"]
