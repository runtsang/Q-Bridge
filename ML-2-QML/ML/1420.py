from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNNGen384(nn.Module):
    """
    A hybrid classical regressor with residual blocks and dropout.
    This architecture is inspired by the original EstimatorQNN but
    extends it with deeper layers, batch normalization, and dropout
    for better generalization on larger datasets.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_blocks: int = 3, dropout: float = 0.2) -> None:
        """
        Args:
            input_dim: dimensionality of the input features.
            hidden_dim: number of hidden units per linear layer.
            num_blocks: number of residual blocks.
            dropout: dropout probability.
        """
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        x = F.relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        return self.output_layer(x)

class ResidualBlock(nn.Module):
    """
    A simple residual block with two linear layers,
    batch normalization, and a skip connection.
    """

    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.bn2(self.linear2(out))
        out = self.dropout(out)
        out += residual
        return F.relu(out)

__all__ = ["EstimatorQNNGen384"]
