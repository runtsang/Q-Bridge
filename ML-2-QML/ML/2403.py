"""Hybrid classical estimator that integrates self‑attention and a simple feed‑forward network."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

# Import the classical self‑attention helper
from.SelfAttention import SelfAttention


def HybridEstimatorQNN():
    """Return a hybrid classical estimator."""

    class HybridEstimator(nn.Module):
        def __init__(self, embed_dim: int = 4, hidden_dim: int = 8) -> None:
            super().__init__()
            # Classical self‑attention block
            self.attention = SelfAttention()(embed_dim=embed_dim)
            # Feed‑forward network
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # Apply classical self‑attention
            attn_output = self.attention.run(
                rotation_params=np.random.randn(embed_dim, embed_dim),
                entangle_params=np.random.randn(embed_dim, embed_dim),
                inputs=inputs.numpy(),
            )
            # Convert back to tensor
            attn_tensor = torch.as_tensor(attn_output, dtype=inputs.dtype, device=inputs.device)
            # Pass through feed‑forward layers
            return self.net(attn_tensor)

    return HybridEstimator()


__all__ = ["HybridEstimatorQNN"]
