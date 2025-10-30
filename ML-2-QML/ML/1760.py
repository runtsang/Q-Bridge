"""
Deep, regularised regression network inspired by the original EstimatorQNN.
Includes batch‑normalisation, dropout and a flexible hidden‑layer stack.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import List, Sequence

class EstimatorQNN(nn.Module):
    """
    Re‑implementation of EstimatorQNN with deeper architecture and regularisation.
    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input features.
    hidden_dims : Sequence[int], default (32, 16)
        Sizes of the hidden layers.
    dropout : float, default 0.1
        Dropout probability applied after each ReLU.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (32, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.net(x)

__all__ = ["EstimatorQNN"]
