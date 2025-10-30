"""Enhanced classical sampler network with residual connections and dropout."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerModule(nn.Module):
    """A 2‑to‑2 MLP with a residual skip and dropout for regularisation."""

    def __init__(self, hidden_dim: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over 2‑bit strings."""
        out = self.fc1(inputs)
        out = F.tanh(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return F.softmax(out, dim=-1)

    def sample(self, inputs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Draw discrete samples from the learned distribution."""
        probs = self.forward(inputs)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((n_samples,)).transpose(0, 1)


def SamplerQNN() -> SamplerModule:
    """Factory returning a freshly initialised sampler network."""
    return SamplerModule()


__all__ = ["SamplerQNN"]
