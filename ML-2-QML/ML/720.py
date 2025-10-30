"""Classical classifier factory with residual, dropout, and batch‑norm layers.

The builder returns a ``torch.nn.Module`` that mirrors the interface of the
quantum helper: a tuple of (network, encoding, weight_sizes, observables).
The network is now a residual block architecture that can be tuned via the
``depth`` parameter.  Dropout and batch‑norm are inserted after every linear
layer to improve generalisation and stabilise training.  The ``encoding`` list
represents the indices of the input features that are used; this is kept for
compatibility with the quantum counterpart.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A single residual block with two linear layers, batch‑norm and ReLU."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        return self.relu(x + residual)


def build_classifier_circuit(
    num_features: int,
    depth: int,
    dropout: float = 0.1,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a residual feed‑forward classifier and metadata.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of residual blocks.
    dropout : float, optional
        Dropout probability applied after the second linear layer in each block.

    Returns
    -------
    network : nn.Module
        The assembled classifier.
    encoding : Iterable[int]
        Indices of features used (identity mapping).
    weight_sizes : Iterable[int]
        Flattened weight counts for each linear layer, useful for parameter
        bookkeeping in hybrid workflows.
    observables : List[int]
        Dummy observable list to keep API parity with the quantum version.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    # Initial linear layer to project into hidden dimension
    init_linear = nn.Linear(in_dim, num_features)
    layers.append(init_linear)
    weight_sizes.append(init_linear.weight.numel() + init_linear.bias.numel())
    layers.append(nn.ReLU())

    # Residual blocks
    for _ in range(depth):
        block = ResidualBlock(num_features, num_features, dropout)
        layers.append(block)
        # Each block has two linear layers
        weight_sizes.extend([block.linear1.weight.numel() + block.linear1.bias.numel(),
                             block.linear2.weight.numel() + block.linear2.bias.numel()])

    # Final head mapping to binary output
    head = nn.Linear(num_features, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # placeholder to satisfy interface

    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
