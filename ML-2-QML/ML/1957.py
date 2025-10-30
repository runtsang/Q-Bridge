"""
Classical sampler network with configurable depth, dropout, and layer normalization.

This advanced sampler extends the original 2‑to‑2 feed‑forward network, allowing arbitrary
input/output dimensionality, multiple hidden layers, and dropout for regularisation. It
produces class probabilities via a softmax activation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedSamplerQNN(nn.Module):
    """
    A flexible neural sampler network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    output_dim : int
        Number of output classes (probabilities to be produced).
    hidden_layers : int
        Number of hidden layers (>=1).
    hidden_dim : int
        Width of each hidden layer.
    dropout : float
        Dropout probability applied after each hidden layer.
    """
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_layers: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class probabilities.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Softmaxed probabilities of shape (..., output_dim).
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)


__all__ = ["AdvancedSamplerQNN"]
