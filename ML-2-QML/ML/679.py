"""EstimatorQNNGen342: an enhanced feed‑forward regressor.

Features:
* 3 hidden layers with 16, 8, and 4 units.
* Batch‑normalisation, ReLU activations, and dropout regularisation.
* Designed for regression on 2‑dimensional inputs.
"""

import torch
from torch import nn

class EstimatorQNNGen342(nn.Module):
    def __init__(self, input_dim: int = 2, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

__all__ = ["EstimatorQNNGen342"]
