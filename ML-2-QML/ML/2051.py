"""Classical classifier that mirrors the interface of the quantum helper.

This module extends the original ``build_classifier_circuit`` by adding
configurable hidden layers, dropout, and a helper to compute the total
parameter count.  The returned network is fully compatible with PyTorch
training loops."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """Builds a feed‑forward classifier with a feature‑encoding layer.

    Parameters
    ----------
    num_features: int
        Dimensionality of the input feature vector.
    depth: int
        Number of hidden layers.
    hidden_dim: int, optional
        Width of each hidden layer.  Defaults to ``num_features``.
    dropout: float, optional
        Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features
        self.dropout = dropout
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        layers: list[nn.Module] = []

        # Encoding / first hidden layer
        layers.append(nn.Linear(self.num_features, self.hidden_dim))
        layers.append(nn.ReLU())

        # Depth of hidden layers
        for _ in range(self.depth):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

        # Output head
        layers.append(nn.Linear(self.hidden_dim, 2))

        return nn.Sequential(*layers)

    def get_parameter_count(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"<QuantumClassifierModel depth={self.depth} "
            f"hidden_dim={self.hidden_dim} dropout={self.dropout}>"
        )


__all__ = ["QuantumClassifierModel"]
