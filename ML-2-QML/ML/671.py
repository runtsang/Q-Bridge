"""Enhanced classical estimator with dropout, batch‑norm, and residual connections."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EstimatorQNN(nn.Module):
    """A robust feed‑forward regressor.

    Features
    --------
    - Two hidden layers with 32 and 16 units.
    - Batch‑normalisation and ReLU activations.
    - Dropout for regularisation.
    - Optional residual skip from input to output.
    """

    def __init__(self, input_dim: int = 2, residual: bool = True) -> None:
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.net(x)
        if self.residual:
            # project to match output shape if necessary
            if x.shape[-1] == 1:
                out = out + x
            else:
                out = out + x[:, :1]
        return out


def EstimatorQNNFactory() -> EstimatorQNN:
    """Return an instance of the upgraded EstimatorQNN."""
    return EstimatorQNN()


__all__ = ["EstimatorQNN", "EstimatorQNNFactory"]
