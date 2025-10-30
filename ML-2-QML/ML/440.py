"""Enhanced classical sampler network with modular architecture and dropout."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen(nn.Module):
    """
    A flexible, dropout‑enabled sampler network.
    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    hidden_dim : int
        Number of units in the hidden layer.
    output_dim : int
        Dimensionality of the output distribution.
    dropout : float
        Drop‑out probability applied after the hidden layer.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a softmax probability distribution.
        """
        return F.softmax(self.net(inputs), dim=-1)

__all__ = ["SamplerQNNGen"]
