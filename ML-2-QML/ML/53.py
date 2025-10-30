"""Enhanced classical estimator with configurable depth and regularisation."""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


def EstimatorQNN(
    input_dim: int = 2,
    hidden_dims: Optional[list[int]] = None,
    dropout: float = 0.1,
    use_batchnorm: bool = True,
) -> nn.Module:
    """
    Return a configurable fully‑connected regression network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : list[int] | None
        Sizes of successive hidden layers. Defaults to [64, 32, 16].
    dropout : float
        Dropout probability applied after each hidden layer.
    use_batchnorm : bool
        Whether to insert a BatchNorm1d after each linear layer.

    Returns
    -------
    nn.Module
        A PyTorch module ready for training with MSE loss.
    """
    if hidden_dims is None:
        hidden_dims = [64, 32, 16]

    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_dim = h_dim

    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


__all__ = ["EstimatorQNN"]
