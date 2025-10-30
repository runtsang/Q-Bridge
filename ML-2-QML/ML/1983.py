"""Enhanced feed‑forward regressor with residual blocks and dropout.

This module defines `EstimatorQNNGen181`, a function that returns a
PyTorch `nn.Module` ready for regression tasks.  The network incorporates
a residual block, dropout for regularisation and a larger hidden
dimension to increase expressive power compared to the original seed.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Simple residual block used in the estimator."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def EstimatorQNNGen181() -> nn.Module:
    """Return a more expressive regression network."""

    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Larger hidden dimension for richer representation
            self.input = nn.Linear(2, 16)
            self.block = ResidualBlock(16)
            self.dropout = nn.Dropout(p=0.2)
            self.output = nn.Linear(16, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            x = self.input(inputs)
            x = self.block(x)
            x = torch.relu(x)
            x = self.dropout(x)
            return self.output(x)

    return EstimatorNN()


__all__ = ["EstimatorQNNGen181"]
