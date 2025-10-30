"""Hybrid classical estimator that also produces parameters for a quantum circuit."""

from __future__ import annotations

import torch
from torch import nn


class HybridQNN(nn.Module):
    """
    A two‑head neural network.
    * regressor: outputs a scalar regression prediction.
    * param_head: outputs two parameters to be fed into a quantum circuit.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.regressor = nn.Linear(hidden_dim, 1)
        self.param_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only the regression output.
        """
        h = self.features(x)
        return self.regressor(h)

    def quantum_params(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate the two parameters that will be used as weight parameters
        in the quantum circuit.  The same feature representation is reused
        to keep the two heads synchronized.
        """
        h = self.features(x)
        return self.param_head(h)


__all__ = ["HybridQNN"]
