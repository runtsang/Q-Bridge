"""Enhanced classical classifier factory mirroring the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(
    num_features: int,
    depth: int,
    dropout: float = 0.1,
    use_batchnorm: bool = True,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a deep feed‑forward classifier with optional dropout and batch‑norm.
    The returned tuple mimics the quantum helper signature so that
    downstream code can treat the two implementations interchangeably.

    Parameters
    ----------
    num_features: int
        Dimensionality of the input feature vector.
    depth: int
        Number of hidden layers.
    dropout: float, optional
        Dropout probability applied after every hidden layer.
    use_batchnorm: bool, optional
        If True, a BatchNorm1d layer follows each linear layer.

    Returns
    -------
    network: nn.Module
        The constructed classifier.
    encoding: Iterable[int]
        Indices of the input features (mirrors the quantum encoding list).
    weight_sizes: Iterable[int]
        Number of trainable parameters in each linear layer.
    observables: List[int]
        Dummy list of observable indices (2 for binary classification).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(num_features))
        layers.append(nn.ReLU(inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # placeholder indices for two binary outputs
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
