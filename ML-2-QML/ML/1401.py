"""Enhanced feed-forward regressor with configurable architecture and regularization."""
from __future__ import annotations

import torch
from torch import nn
from typing import Iterable

def EstimatorQNN(
    input_dim: int = 2,
    hidden_dims: Iterable[int] = (8, 4),
    activation: nn.Module = nn.Tanh(),
    dropout_prob: float = 0.0,
    bias: bool = True,
) -> nn.Module:
    """
    Construct a fully‑connected regression network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : Iterable[int]
        Sequence of hidden layer widths.
    activation : nn.Module
        Activation function applied after each hidden layer.
    dropout_prob : float
        Drop‑out probability applied after each hidden layer (0.0 disables).
    bias : bool
        Whether to include bias terms in linear layers.

    Returns
    -------
    nn.Module
        Instantiated network ready for training.
    """
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim, bias=bias))
        layers.append(activation)
        if dropout_prob > 0.0:
            layers.append(nn.Dropout(dropout_prob))
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, 1, bias=bias))
    return nn.Sequential(*layers)

__all__ = ["EstimatorQNN"]
