"""Advanced classical regressor with residuals and dropout."""
from __future__ import annotations

import torch
from torch import nn


class AdvancedEstimatorQNN(nn.Module):
    """
    A deeper feed‑forward network with residual connections and dropout.
    Designed to mirror the original EstimatorQNN while providing richer
    representational capacity for regression tasks.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [16, 8, 4]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Residual mapping from input to first hidden layer
        self.residual = nn.Linear(input_dim, hidden_dims[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Add a residual connection before the main network
        res = self.residual(x)
        return self.net(res)


def EstimatorQNN() -> AdvancedEstimatorQNN:
    """Return an instance of the advanced classical regressor."""
    return AdvancedEstimatorQNN()


__all__ = ["EstimatorQNN", "AdvancedEstimatorQNN"]
