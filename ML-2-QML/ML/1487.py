"""Enhanced classical feed‑forward regressor with optional regularization and dynamic architecture.

The model accepts an arbitrary input dimension and number of hidden layers,
providing dropout and batch‑normalisation for robust training.
"""

from __future__ import annotations

import torch
from torch import nn


def EstimatorQNN(
    input_dim: int = 2,
    hidden_sizes: tuple[int,...] = (32, 16),
    dropout: float = 0.1,
    use_batchnorm: bool = True,
    activation: nn.Module = nn.ReLU(),
) -> nn.Module:
    """
    Construct a fully‑connected regression network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_sizes : tuple[int,...]
        Sequence of hidden layer sizes.
    dropout : float
        Dropout probability applied after each hidden layer.
    use_batchnorm : bool
        Whether to insert a BatchNorm1d after each hidden layer.
    activation : nn.Module
        Activation function to use after each hidden layer.
    """
    layers = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(activation)
        layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))  # regression output
    return nn.Sequential(*layers)


__all__ = ["EstimatorQNN"]
