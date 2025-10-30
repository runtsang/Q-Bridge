"""Enhanced classical classifier factory with dropout, batch normalization, and flexible hidden layers.

The API mirrors the quantum variant, returning a network and metadata for later use in hybrid training pipelines.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import torch
import torch.nn as nn


def build_classifier_circuit(
    num_features: int,
    hidden_dims: Sequence[int] = (64, 32),
    dropout: float = 0.0,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier.

    Parameters
    ----------
    num_features : int
        Input dimensionality.
    hidden_dims : Sequence[int], optional
        Sizes of the hidden layers. Defaults to (64, 32).
    dropout : float, optional
        Dropout probability applied after each hidden layer. 0.0 disables dropout.

    Returns
    -------
    network : nn.Module
        Sequential model ready for training.
    encoding : Iterable[int]
        Indices of the input features (used for hybrid pipelines).
    weight_sizes : Iterable[int]
        Number of trainable parameters per linear layer, useful for bookkeeping.
    observables : list[int]
        Dummy observable indices, matching the quantum interface.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    weight_sizes: list[int] = []

    # Input to first hidden layer
    for dim in hidden_dims:
        linear = nn.Linear(in_dim, dim)
        layers.append(linear)
        layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = dim

    # Output layer
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = list(range(2))  # placeholder for compatibility
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
