"""Enhanced fully‑connected regressor with residual connections and dropout.

This module defines EstimatorQNN, a PyTorch neural network that extends the original
tiny model by adding depth, batch‑normalization, and dropout for improved generalisation.
"""

import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """Deep residual regressor with dropout."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = h + self.res_block1(h)
        h = h + self.res_block2(h)
        return self.output_layer(h)

__all__ = ["EstimatorQNN"]
