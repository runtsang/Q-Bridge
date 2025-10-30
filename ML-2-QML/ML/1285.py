"""Enhanced classical sampler network with deeper architecture and sampling utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SamplerQNNExtended(nn.Module):
    """
    A richer sampler network:
    - 4 hidden layers with dropout for regularisation.
    - Output probabilities via softmax.
    - `sample` method for stochastic sampling.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: Tuple[int,...] = (8, 16, 8), output_dim: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        return F.softmax(self.net(inputs), dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the network output.
        """
        probs = self.forward(inputs)
        batch_size = probs.shape[0]
        probs_exp = probs.unsqueeze(1).expand(-1, num_samples, -1)
        probs_flat = probs_exp.reshape(-1, probs.shape[-1])
        samples = torch.multinomial(probs_flat, 1).squeeze(-1)
        return samples.reshape(batch_size, num_samples)

def SamplerQNN() -> SamplerQNNExtended:
    """Return an instance of the extended sampler."""
    return SamplerQNNExtended()

__all__ = ["SamplerQNN", "SamplerQNNExtended"]
