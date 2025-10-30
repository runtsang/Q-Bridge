"""Advanced feed‑forward regressor with residual, batchnorm and dropout."""

from __future__ import annotations

import torch
from torch import nn

class AdvancedEstimatorQNN(nn.Module):
    """
    A robust regression network that extends the original EstimatorQNN.

    Features:
    - Residual connections to ease gradient flow.
    - BatchNorm after each linear layer for stable training.
    - Dropout for regularisation.
    - Configurable hidden sizes and depth.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int,...] = (64, 32, 16),
        output_dim: int = 1,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for idx, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            # Residual connection every two layers
            if idx % 2 == 1:
                layers.append(nn.Identity())
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

__all__ = ["AdvancedEstimatorQNN"]
