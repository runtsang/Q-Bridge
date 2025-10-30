"""Enhanced classical classifier mirroring quantum interface with residuals and regularization."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """Feed‑forward network with residual connections, batch‑norm and optional dropout."""

    def __init__(self, num_features: int, depth: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.depth = depth
        self.dropout = dropout

        layers: List[nn.Module] = []
        in_dim = num_features
        self.encoding = list(range(num_features))
        self.weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            bn = nn.BatchNorm1d(num_features)
            layers.extend([linear, bn, nn.ReLU()])
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())

        if dropout > 0.0:
            layers.insert(0, nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self.observables = [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def get_metadata(self) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """Return the underlying network, encoding, weight sizes, and observables."""
        return self.network, self.encoding, self.weight_sizes, self.observables


def build_classifier_circuit(num_features: int, depth: int = 3, dropout: float = 0.0) -> Tuple[QuantumClassifierModel, List[int], List[int], List[int]]:
    """Construct a feed-forward classifier and metadata similar to the quantum variant."""
    model = QuantumClassifierModel(num_features, depth, dropout)
    return model, model.encoding, model.weight_sizes, model.observables


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
