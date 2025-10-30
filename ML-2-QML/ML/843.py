"""Enhanced feed-forward regression network with residual connections and dropout."""

import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNN(nn.Module):
    """A deeper fully‑connected network with residual connections and dropout.

    The architecture mirrors the original two‑layer network but adds:
    - Two hidden layers with 32 units each.
    - BatchNorm and ReLU activations.
    - Dropout for regularisation.
    - Residual skip connection from input to the second hidden layer.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        out = F.relu(self.bn1(self.input_layer(x)))
        residual = out
        out = F.relu(self.bn2(self.hidden_layer(out)))
        out = out + residual  # residual skip
        out = self.dropout(out)
        return self.output_layer(out)

__all__ = ["EstimatorQNN"]
