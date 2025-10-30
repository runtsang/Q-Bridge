from __future__ import annotations

import torch
from torch import nn


class EstimatorQNN(nn.Module):
    """A deeper, regularised feed‑forward regressor.
    It includes LayerNorm, ReLU, and Dropout for improved generalisation.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | tuple[int,...] = (32, 16, 8),
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(x)


__all__ = ["EstimatorQNN"]
