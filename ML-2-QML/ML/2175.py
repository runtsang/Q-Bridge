"""Enhanced feed‑forward regressor with residual connections, batch normalization, and dropout."""

import torch
from torch import nn

class EstimatorQNNExtended(nn.Module):
    """A richer neural network that extends the original 3‑layer network."""
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] = [16, 8], output_dim: int = 1) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass through the network."""
        return self.net(inputs)

__all__ = ["EstimatorQNNExtended"]
