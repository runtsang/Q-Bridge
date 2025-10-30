"""Extended feed‑forward regressor with dropout, batch‑norm and residual connections."""
from __future__ import annotations

import torch
from torch import nn


class _EstimatorQNN(nn.Module):
    """A richer neural network for regression.

    Architecture:
        - Input layer: 2 features
        - 4 hidden layers with increasing dimension (8 → 16 → 8 → 4)
        - Each hidden layer has BatchNorm, ReLU and Dropout(0.2)
        - Residual connection added after the second layer
        - Output layer: single neuron with linear activation

    The design improves expressiveness while mitigating overfitting
    through regularisation and residual pathways.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, 8),
                    nn.BatchNorm1d(8),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ),
                nn.Sequential(
                    nn.Linear(8, 16),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ),
                nn.Sequential(
                    nn.Linear(16, 8),
                    nn.BatchNorm1d(8),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ),
                nn.Sequential(
                    nn.Linear(8, 4),
                    nn.BatchNorm1d(4),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ),
                nn.Linear(4, 1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Residual connection between input and second layer
        residual = x
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = x + residual
        for layer in self.layers[2:]:
            x = layer(x)
        return x


def EstimatorQNN() -> _EstimatorQNN:
    """Return an instance of the extended regressor."""
    return _EstimatorQNN()


__all__ = ["EstimatorQNN"]
