"""Enhanced feed‑forward regressor with regularisation and a deeper architecture.

The model expands the original 2‑layer network to a 3‑layer architecture with
batch‑normalisation and dropout.  This makes the network more expressive
while providing a simple regularisation pipeline that can be reused in
downstream experiments.
"""

import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """
    A small but regularised regression network.

    Architecture
    ------------
    - Linear(2, 32) → BatchNorm1d → ReLU → Dropout(0.1)
    - Linear(32, 16) → BatchNorm1d → ReLU → Dropout(0.1)
    - Linear(16, 1)

    The network is fully differentiable and can be trained with any
    optimiser from torch.optim.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

def estimator_qnn() -> EstimatorQNN:
    """Return an instance of the regularised regression network."""
    return EstimatorQNN()

__all__ = ["EstimatorQNN", "estimator_qnn"]
