"""Enhanced classical classifier mirroring the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """
    A robust feed‑forward classifier with optional regularisation and a
    configurable depth.  Mirrors the original `build_classifier_circuit`
    signature but adds batch‑norm, dropout and a deeper head for better
    generalisation on high‑dimensional tabular data.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim

        layers: list[nn.Module] = []
        in_dim = num_features

        # Encoding block – linear + batch‑norm + ReLU
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Deep variational block: repeated linear–ReLU layers
        for _ in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Classification head
        layers.append(nn.Linear(hidden_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int = 4,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """
        Return a network and metadata compatible with the quantum helper.
        The metadata encodes the feature indices, weight counts per layer,
        and observable indices used for comparison in hybrid experiments.
        """
        layers: list[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: list[int] = []

        # Encode input
        layers.append(nn.Linear(in_dim, num_features))
        weight_sizes.append(layers[-1].weight.numel() + layers[-1].bias.numel())

        # Variational layers
        for _ in range(depth):
            linear = nn.Linear(num_features, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())

        # Output head
        head = nn.Linear(num_features, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
