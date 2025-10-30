"""Extended classical regressor with dropout, weight decay, and custom initialization."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EstimatorQNN(nn.Module):
    """A deeper, regularized feed‑forward network for regression.

    The architecture consists of an input layer, two hidden layers with
    ReLU activations, dropout for robustness, and a final linear output.
    L2 weight decay is applied during training, and the weights are
    initialized using He normal initialization for better convergence.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        # He normal initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


__all__ = ["EstimatorQNN"]
