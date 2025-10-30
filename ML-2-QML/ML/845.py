"""Enhanced classical feed-forward regressor with residual connections, batch norm, and dropout.

This class extends the original simple two-layer network by adding:
- multiple hidden layers with batch normalization
- dropout for regularisation
- a residual path from input to hidden layers
- optional hyperparameters for hidden size and dropout probability
"""

import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """Classical neural network for regression."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Residual mapping
        self.residual = nn.Linear(input_dim, hidden_dim)

        # Hidden layers
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        res = self.residual(x)
        out = self.layer1(x)
        out = self.layer2(out + res)
        out = self.layer3(out)
        return self.output(out)

__all__ = ["EstimatorQNN"]
