"""Enhanced classical classifier with modular architecture and optional regularization.

The class mirrors the quantum helper interface but augments it with dropout,
batch‑normalisation and a flexible depth parameter.  A static
``build_classifier_circuit`` method remains for compatibility with the
original seed, yet it now returns the layer sizes and parameter counts in
a more informative form.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifier(nn.Module):
    """Feed‑forward neural network that can be used as a drop‑in replacement
    for the quantum classifier in the original repository.
    """
    def __init__(self, num_features: int, depth: int = 2,
                 dropout: float = 0.0, use_batchnorm: bool = False) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def build_classifier_circuit(num_features: int,
                                 depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Return a lightweight descriptor of the network architecture.
        ``encoding`` now contains the indices of the input features that
        feed into the first layer, ``weight_sizes`` lists the number of
        trainable parameters per layer, and ``observables`` is a placeholder
        list that preserves the interface used by the quantum version.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifier"]
