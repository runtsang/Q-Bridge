"""Enhanced classical estimator with deeper network and regularization.

The model is a fully‑connected regressor that extends the original
two‑layer design.  Hidden layers are 16‑32‑16 units, each followed by
BatchNorm1d and ReLU, with dropout to mitigate over‑fitting.  The
architecture is still compact enough for quick prototyping but
captures richer feature interactions.

Usage:
    >>> from EstimatorQNN__gen312 import EstimatorQNN
    >>> model = EstimatorQNN()
    >>> output = model(torch.rand(5, 2))
"""

import torch
from torch import nn
from torch.nn import functional as F

class EstimatorQNN(nn.Module):
    """Deep feed‑forward regression network with batch‑norm and dropout."""
    def __init__(self, input_dim: int = 2, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

__all__ = ["EstimatorQNN"]
