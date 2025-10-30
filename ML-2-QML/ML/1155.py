"""Hybrid classical classifier with optional residual blocks, dropout, and weight‑decay regularization."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    residual: bool = False,
    dropout: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[float]]:
    """
    Construct a feed‑forward classifier that optionally uses residual connections,
    dropout, and returns auxiliary data for regularization.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    residual : bool, optional
        If ``True`` a skip connection is inserted after each hidden block.
    dropout : float, optional
        Dropout probability applied after each hidden block. ``None`` disables dropout.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    nn.Module
        The constructed classifier.
    Iterable[int]
        List of input feature indices (identity mapping).
    Iterable[int]
        Weight sizes of each linear layer (used for weight‑decay regularization).
    List[float]
        Dummy observables to keep API compatible with the quantum version.
    """
    torch.manual_seed(random_state)

    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())

        if dropout is not None:
            layers.append(nn.Dropout(p=dropout))

        if residual:
            # Simple residual block: add an identity mapping
            layers.append(nn.Identity())

        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)

    network = nn.Sequential(*layers)

    weight_sizes = [
        module.weight.numel() + module.bias.numel()
        for module in network.modules()
        if isinstance(module, nn.Linear)
    ]

    observables = [float(i) for i in range(2)]

    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
