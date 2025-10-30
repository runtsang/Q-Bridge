"""Classical classifier factory mirroring the quantum helper interface with enhanced layers."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier with optional hidden dimension, batch‑norm,
    and dropout. The returned values are kept compatible with the quantum API:
    * ``model`` – a :class:`torch.nn.Sequential` ready for training.
    * ``encoding`` – indices of the input features (used by the quantum wrapper).
    * ``weight_sizes`` – number of trainable parameters per layer.
    * ``observables`` – dummy observable indices used by the quantum interface.

    Parameters
    ----------
    num_features : int
        Number of input features / qubits.
    depth : int
        Number of hidden layers.
    hidden_dim : int | None, optional
        Size of each hidden layer. Defaults to ``num_features``.
    dropout : float, optional
        Dropout probability applied after every hidden layer.

    Returns
    -------
    Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]
    """
    layers: List[nn.Module] = []

    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    hidden_dim = hidden_dim or num_features

    for _ in range(depth):
        linear = nn.Linear(in_dim, hidden_dim)
        layers.append(linear)
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = hidden_dim

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
