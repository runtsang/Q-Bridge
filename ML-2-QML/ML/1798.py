"""Enhanced classical classifier with optional dropout, spectral normalization,
and layer‑specific weight statistics for hybrid experimentation."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


class QuantumClassifierModel(nn.Module):
    """Drop‑in replacement for the original build_classifier_circuit factory.

    The network consists of `depth` hidden layers that optionally include
    dropout and spectral‑norm regularization.  The constructor mirrors the
    signature of the seed function, but returns a ``nn.Module`` instead
    of a tuple.  A ``get_metadata`` method exposes the same encoding,
    weight size, and observable list so that downstream QML code can
    remain compatible.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        dropout_rate: float = 0.0,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_features

        # Encode the feature indices – identical to the seed
        self.encoding: list[int] = list(range(num_features))
        self.weight_sizes: list[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.ReLU())
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

        head = nn.Linear(in_dim, 2)
        if spectral_norm:
            head = nn.utils.spectral_norm(head)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())

        self.network = nn.Sequential(*layers)

        # Placeholder observables to preserve API compatibility
        self.observables: list[torch.Tensor] = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.network(x)

    def get_metadata(
        self,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[torch.Tensor]]:
        """Return the network and metadata matching the seed's tuple."""
        return self.network, self.encoding, self.weight_sizes, self.observables
