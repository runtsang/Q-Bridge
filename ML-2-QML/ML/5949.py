"""Enhanced classical neural network for regression with feature mapping and dropout."""
from __future__ import annotations

import torch
from torch import nn

def EstimatorQNN() -> nn.Module:
    """
    Return an advanced regression network that expands the input feature space,
    applies dropout for regularisation, and uses a deeper feed‑forward backbone.
    The network is compatible with the original EstimatorQNN interface.
    """
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Feature‑mapping layer: 2 → 8 (captures interactions)
            self.feature_map = nn.Linear(2, 8, bias=False)
            # Main network with dropout after the first hidden layer
            self.net = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(16, 8),
                nn.Tanh(),
                nn.Linear(8, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            """
            Forward pass.

            Parameters
            ----------
            inputs : torch.Tensor
                Tensor of shape (batch_size, 2) containing the two input features.

            Returns
            -------
            torch.Tensor
                Tensor of shape (batch_size, 1) with the regression output.
            """
            mapped = self.feature_map(inputs)
            return self.net(mapped)

    return EstimatorNN()


__all__ = ["EstimatorQNN"]
