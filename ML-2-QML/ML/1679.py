"""Enhanced classical classifier factory with residual blocks and configurable depth."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A simple residual block with optional linear projection for dimension mismatch."""
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x) + self.proj(x))


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dim: int | None = None,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a deep residual classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of residual blocks to stack.
    hidden_dim : int | None, optional
        Width of the hidden layers; defaults to ``num_features`` for a square network.

    Returns
    -------
    network : nn.Module
        The constructed Torch model.
    encoding : Iterable[int]
        Indices of the input features that are fed into the network.
    weight_sizes : Iterable[int]
        Total number of trainable parameters in each linear layer (including bias).
    observables : List[int]
        Indices of the output logits (always 0 and 1 for binary classification).
    """
    hidden_dim = hidden_dim or num_features
    layers: List[nn.Module] = []

    # Input projection
    layers.append(nn.Linear(num_features, hidden_dim))
    layers.append(nn.ReLU())

    # Residual blocks
    for _ in range(depth):
        layers.append(ResidualBlock(hidden_dim, hidden_dim))

    # Output head
    head = nn.Linear(hidden_dim, 2)
    layers.append(head)

    network = nn.Sequential(*layers)

    # Compute metadata
    weight_sizes = []
    for module in network:
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())

    encoding = list(range(num_features))
    observables = [0, 1]
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
