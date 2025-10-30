"""Enhanced classical classifier factory with configurable hidden layers and dropout."""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import torch
import torch.nn as nn


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_units: Sequence[int] | None = None,
    dropout: float = 0.0,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward neural network that mirrors the quantum helper
    interface but with optional hidden‑layer customization.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input data.
    depth : int
        Number of hidden layers to stack.
    hidden_units : Sequence[int] | None, optional
        Size of each hidden layer.  If ``None`` the network uses ``num_features``
        units for every layer.  The sequence length must match ``depth``.
    dropout : float, optional
        Dropout probability applied after each hidden layer.  A value of 0.0
        disables dropout.

    Returns
    -------
    nn.Module
        A ``torch.nn.Sequential`` network.
    Iterable[int]
        Indices of the input features used for encoding (identity mapping).
    Iterable[int]
        Number of trainable parameters per linear layer, including bias.
    list[int]
        Dummy observable indices compatible with the quantum API.
    """
    if hidden_units is None:
        hidden_units = [num_features] * depth
    if len(hidden_units)!= depth:
        raise ValueError("hidden_units length must equal depth")

    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for units in hidden_units:
        linear = nn.Linear(in_dim, units)
        layers.append(linear)
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = units

    # Final classification head
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
