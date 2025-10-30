"""
Enhanced feed‑forward regressor with batch‑normalisation, dropout and a residual style architecture.

Usage:
    model = EstimatorQNNGen033()
"""

from __future__ import annotations

import torch
from torch import nn

def EstimatorQNNGen033() -> nn.Module:
    """
    Return a robust regression network suitable for small tabular data.
    The architecture contains three hidden layers, batch‑normalisation,
    ReLU activations, dropout and a final linear output.
    """
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()
