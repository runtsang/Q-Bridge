"""Enhanced classical classifier with configurable depth, hidden size, and output classes.

The function returns:
    - model: nn.Sequential network
    - encoding: indices of input features (used by the quantum counterpart)
    - weight_sizes: list of number of parameters in each linear layer
    - observables: dummy list of indices (class labels) to keep API symmetry
"""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(
    num_features: int,
    hidden_size: int = 64,
    depth: int = 2,
    num_classes: int = 3,
    dropout_prob: float = 0.0,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier and metadata that mirror the quantum interface.

    Parameters
    ----------
    num_features : int
        Number of input features (dimension of the data).
    hidden_size : int, default 64
        Width of each hidden layer.
    depth : int, default 2
        Number of hidden layers.
    num_classes : int, default 3
        Number of output classes.
    dropout_prob : float, default 0.0
        Dropout probability applied after each hidden layer (if > 0).

    Returns
    -------
    Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]
        * model : nn.Sequential
            The constructed neural network.
        * encoding : Iterable[int]
            Feature indices, matching the quantum encoding scheme.
        * weight_sizes : Iterable[int]
            Number of parameters in each linear layer.
        * observables : List[int]
            Dummy observable indices (one per class) to keep API symmetry.
    """
    layers: List[nn.Module] = []

    # Input layer
    layers.append(nn.Linear(num_features, hidden_size))
    layers.append(nn.ReLU())
    if dropout_prob > 0.0:
        layers.append(nn.Dropout(dropout_prob))

    weight_sizes = [layers[0].weight.numel() + layers[0].bias.numel()]

    # Hidden layers
    for _ in range(depth - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        if dropout_prob > 0.0:
            layers.append(nn.Dropout(dropout_prob))
        weight_sizes.append(layers[-4].weight.numel() + layers[-4].bias.numel())

    # Output layer
    layers.append(nn.Linear(hidden_size, num_classes))
    weight_sizes.append(layers[-2].weight.numel() + layers[-2].bias.numel())

    model = nn.Sequential(*layers)

    encoding = list(range(num_features))
    observables = list(range(num_classes))

    return model, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
