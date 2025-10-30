"""Extended classical sampler network with dropout, batch‑norm and a training helper."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtendedSamplerQNN(nn.Module):
    """
    A richer sampler that mirrors the original architecture but adds
    dropout and batch‑norm for better generalisation.  It also
    exposes a ``predict`` helper that returns class probabilities.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8,
                 output_dim: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        return self.forward(x)


__all__ = ["ExtendedSamplerQNN"]
