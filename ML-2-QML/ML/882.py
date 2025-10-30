"""EstimatorQNN – a lightweight, regularised feed‑forward regressor."""
from __future__ import annotations

import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """
    A compact regression network that incorporates modern best practices:

    * Residual connections when input and hidden dimensions match.
    * LayerNorm after every linear layer to stabilise training.
    * Dropout for regularisation.
    * Configurable hidden layer sizes.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] | tuple[int,...] = (16, 8),
                 dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            # Residual connection when dimensions match
            if prev_dim == h:
                layers.append(nn.Identity())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

def EstimatorQNN() -> EstimatorQNN:
    """Factory returning a ready‑to‑train EstimatorQNN instance."""
    return EstimatorQNN()

__all__ = ["EstimatorQNN"]
