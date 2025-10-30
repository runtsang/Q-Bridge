"""EstimatorQNNModel – classical residual network with batch‑norm and dropout.

The module implements a deep feed‑forward regressor that can be trained with
PyTorch optimisers.  It replaces the original 3‑layer network with a stack of
residual blocks, each featuring a linear layer, batch‑normalisation, ReLU,
dropout and a skip connection.  The design improves gradient flow and
regularisation while keeping the API identical to the seed.

Usage
-----
>>> from EstimatorQNN__gen252 import EstimatorQNN
>>> model = EstimatorQNN()
>>> loss_fn = torch.nn.MSELoss()
>>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["EstimatorQNN"]


class ResidualBlock(nn.Module):
    """A single residual block with batch‑norm, ReLU and dropout."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(p=0.2)
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return F.relu(out + self.shortcut(x))


class EstimatorNN(nn.Module):
    """Residual feed‑forward network for regression.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of input features.
    hidden_dims : Sequence[int], default (64, 32, 16)
        Sizes of the hidden residual blocks.
    output_dim : int, default 1
        Size of the regression output.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int,...] = (64, 32, 16), output_dim: int = 1) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(ResidualBlock(prev_dim, h))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def EstimatorQNN() -> EstimatorNN:
    """Factory returning a pre‑configured residual regressor."""
    return EstimatorNN()
