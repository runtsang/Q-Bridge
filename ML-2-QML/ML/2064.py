"""Enhanced classical sampler network with residual connections and dropout.

This module defines SamplerQNN, an improved neural network that extends the
original two‑layer architecture. It supports arbitrary input dimension,
configurable hidden sizes, dropout, and a residual skip connection
between the first and second hidden layers. The output is a probability
distribution over the two target classes.

The network can be used as a drop‑in replacement for the original
SamplerQNN while providing richer expressive power for downstream
training pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """Classical sampler with residual and dropout.

    Parameters
    ----------
    in_features : int
        Number of input features (default 2).
    hidden_features : int
        Size of the hidden layer (default 8).
    dropout : float
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.dropout = dropout

        # Primary linear layer
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Residual pathway
        self.residual = nn.Linear(in_features, hidden_features)
        # Output linear layer
        self.fc2 = nn.Linear(hidden_features, 2)

        # Dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over two classes."""
        # Primary path
        x = F.relu(self.fc1(inputs))
        # Residual path
        res = self.residual(inputs)
        # Combine with skip connection
        x = x + res
        x = self.drop(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


__all__ = ["SamplerQNN"]
