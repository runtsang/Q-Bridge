"""Classical classifier factory with dropout and regularization."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier with dropout and L2 regularization metadata.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.

    Returns
    -------
    network : nn.Module
        Sequential network consisting of Linear → ReLU → Dropout layers, ending with a Linear head.
    encoding : Iterable[int]
        Identity mapping from input indices to network input positions.
    weight_sizes : Iterable[int]
        Number of trainable parameters per layer (weights + biases).
    observables : list[int]
        Integer labels for the two-class output (mirrors the quantum observable indices).
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    # Build hidden layers with dropout
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU(), nn.Dropout(p=0.1)])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    # Output head
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # binary classification labels

    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
