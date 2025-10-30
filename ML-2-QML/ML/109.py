"""Classical classifier factory mirroring the quantum helper interface with advanced features."""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class QuantumClassifierModel(nn.Module):
    """
    A flexible feed‑forward classifier that mirrors the interface of the quantum factory.
    Supports optional dropout, batch‑normalisation and an arbitrary number of hidden layers.
    """

    def __init__(self, num_features: int, depth: int = 2, dropout: float = 0.0, batchnorm: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features

        # Feature encoding indices (identity mapping)
        self._encoding = list(range(num_features))

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            if batchnorm:
                layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features

        # Output head
        self._head = nn.Linear(in_dim, 2)
        layers.append(self._head)

        self.network = nn.Sequential(*layers)

        # Observables (class labels)
        self._observables = list(range(2))

        # Store weight sizes for introspection
        self._weight_sizes = [m.weight.numel() + m.bias.numel() for m in self.modules() if isinstance(m, nn.Linear)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""
        return self.network(x)

    @property
    def encoding(self) -> Iterable[int]:
        """Indices used for feature encoding."""
        return self._encoding

    @property
    def weight_sizes(self) -> Iterable[int]:
        """Number of trainable parameters per linear layer."""
        return self._weight_sizes

    @property
    def observables(self) -> Iterable[int]:
        """Class labels represented as observables."""
        return self._observables

    def get_state_dict(self):
        """Convenience wrapper for the underlying state dict."""
        return self.network.state_dict()


def build_classifier_circuit(num_features: int, depth: int, dropout: float = 0.0, batchnorm: bool = False) -> QuantumClassifierModel:
    return QuantumClassifierModel(num_features, depth, dropout, batchnorm)


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
