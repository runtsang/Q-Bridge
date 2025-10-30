"""Classical classifier factory mirroring the quantum helper interface, extended with residual blocks and regularisation."""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A simple residual block with optional batch‑norm and dropout."""
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        out = torch.relu(out)
        out = self.fc(out)
        out = self.dropout(out)
        return x + out  # residual connection


def build_classifier_circuit(num_features: int, depth: int,
                             hidden_dim: int | None = None,
                             dropout: float = 0.0) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier with optional residual connections.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    depth : int
        Number of hidden stages (each stage may contain a residual block).
    hidden_dim : int | None
        Size of hidden layers.  If ``None`` the input dimensionality is reused.
    dropout : float
        Drop‑out probability applied inside each residual block.

    Returns
    -------
    network : nn.Module
        A sequential model ready for training.
    encoding : Iterable[int]
        A placeholder list of feature indices that mirror the quantum interface.
    weight_sizes : Iterable[int]
        Total number of learnable parameters per layer (flattened).
    observables : list[int]
        Indices used by the quantum wrapper to map to output logits (kept identical for API compatibility).
    """
    hidden_dim = hidden_dim or num_features
    layers: list[nn.Module] = []

    # Input layer
    layers.append(nn.Linear(num_features, hidden_dim))
    layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.ReLU())

    # Residual blocks
    for _ in range(depth):
        layers.append(ResidualBlock(hidden_dim, dropout))

    # Output head
    layers.append(nn.Linear(hidden_dim, 2))  # binary classification
    network = nn.Sequential(*layers)

    # Compute per‑layer weight counts (including biases)
    weight_sizes = [p.numel() for p in network.parameters()]

    # Encoding placeholder mirrors quantum encoding indices
    encoding = list(range(num_features))

    # Observables placeholder for compatibility (class indices)
    observables = [0, 1]

    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
