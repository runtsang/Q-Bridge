"""
Classical feed‑forward classifier builder that mirrors the quantum circuit API.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    use_bias: bool = True,
    dropout: float = 0.0,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a depth‑controlled feed‑forward network that returns the same
    metadata layout as the quantum builder.

    Parameters
    ----------
    num_features : int
        Number of input features / qubits.
    depth : int
        Number of hidden layers; each expands to ``num_features`` units.
    use_bias : bool
        Whether each linear layer has a bias term.
    dropout : float
        Dropout probability after each ReLU.

    Returns
    -------
    network : nn.Module
        ``nn.Sequential`` of Linear → ReLU stacks followed by a linear head.
    encoding : Iterable[int]
        Index list that mirrors the quantum encoding vector.
    weight_sizes : Iterable[int]
        Number of trainable parameters per linear layer.
    observables : List[int]
        Dummy observable indices matching the quantum output shape (2).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features, bias=use_bias)
        layers.append(linear)
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        weight_sizes.append(
            linear.weight.numel() + (linear.bias.numel() if use_bias else 0)
        )
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)

    # Observables are placeholders for Pauli‑Z measurements on each qubit
    observables = [0, 1]
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
