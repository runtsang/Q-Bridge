"""Enhanced classical sampler network with residual connections, batch normalization, and dropout."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block with linear layers, batchnorm, and ReLU."""
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return self.relu(out + x)

def EnhancedSamplerQNN(hidden_dim: int = 8, dropout: float = 0.1) -> nn.Module:
    """
    Construct an enhanced sampler network with:
      * 2 hidden layers + residual block
      * batch normalization and dropout
      * softmax output for probability distribution
    """
    class EnhancedModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                ResidualBlock(hidden_dim, dropout),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Return probability distribution over 2 classes."""
            return F.softmax(self.net(inputs), dim=-1)

    return EnhancedModule()
