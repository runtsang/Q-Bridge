from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HybridLayer(nn.Module):
    """
    HybridLayer implements a classical fully connected network that optionally
    incorporates a sampler network. It mirrors the structure of the original
    FCL and SamplerQNN seeds while adding configurable depth and dropout for
    improved scaling.
    """

    def __init__(self, n_features: int = 1, hidden_dim: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc = nn.Linear(n_features, hidden_dim)
        self.sampler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies a linear transformation followed by a
        sampler network, ending with a softmax over the two output classes.
        """
        x = self.dropout(self.fc(x))
        probs = F.softmax(self.sampler(x), dim=-1)
        return probs


__all__ = ["HybridLayer"]
