"""Enhanced classical classifier factory with dropout and batch‑norm support."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward neural network that mirrors the interface of the quantum helper.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    dropout : float, optional
        Dropout probability applied after each hidden layer. 0.0 disables dropout.
    batch_norm : bool, optional
        Whether to insert a BatchNorm1d layer after each hidden layer.

    Returns
    -------
    network : nn.Module
        The constructed sequential model.
    encoding : Iterable[int]
        Identity mapping of input features.
    weight_sizes : Iterable[int]
        Number of trainable parameters per layer.
    observables : list[int]
        Class indices (0 and 1 for binary classification).
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())

        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features))
        layers.append(nn.ReLU())

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
