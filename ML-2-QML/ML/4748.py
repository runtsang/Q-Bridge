"""Hybrid classical classifier mirroring the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(
    num_features: int,
    depth: int,
    dropout_prob: float = 0.0,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier and expose metadata identical to the quantum variant.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.  Each layer maps `num_features -> num_features`.
    dropout_prob : float, optional
        Dropout probability after every ReLU.  Defaults to 0.0 (no dropout).

    Returns
    -------
    network : nn.Sequential
        The classical classifier.
    encoding : Iterable[int]
        Indices of the input features that are encoded into the quantum circuit.
    weight_sizes : Iterable[int]
        Number of trainable parameters for every layer, in order.
    observables : List[int]
        Dummy list matching the shape of the quantum observables (used for API parity).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        if dropout_prob > 0.0:
            layers.append(nn.Dropout(dropout_prob))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)  # binary classification
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))  # placeholder to keep signature identical
    return network, encoding, weight_sizes, observables


class HybridClassifier(nn.Module):
    """
    Classical classifier that can be dropped into pipelines that expect the
    quantum helper interface.  It simply wraps the network produced by
    :func:`build_classifier_circuit`.

    The class exposes the same public attributes as the quantum counterpart:
    ``encoding``, ``weight_sizes`` and ``observables`` for consistency.
    """

    def __init__(self, num_features: int, depth: int = 4, dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth, dropout_prob
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

__all__ = ["HybridClassifier", "build_classifier_circuit"]
