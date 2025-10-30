"""
Enhanced classical sampler network.

This module defines `SamplerQNN`, a PyTorch neural network that
mirrors the original two‑layer design but adds depth, dropout,
and batch‑normalisation to improve generalisation.  The
`sample` method returns a soft‑max probability vector suitable
for downstream tasks such as reinforcement‑learning policies
or probabilistic inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    A deeper, regularised sampler network.

    Architecture:
        Linear(2 → 8) → BatchNorm1d → ReLU → Dropout(0.2)
        Linear(8 → 16) → BatchNorm1d → ReLU → Dropout(0.2)
        Linear(16 → 8) → BatchNorm1d → ReLU
        Linear(8 → 2) → Softmax
    """

    def __init__(self, dropout: float = 0.2, seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Linear(8, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over the two output classes."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that returns a sampled one‑hot vector
        drawn from the soft‑max distribution.
        """
        probs = self.forward(inputs)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

__all__ = ["SamplerQNN"]
