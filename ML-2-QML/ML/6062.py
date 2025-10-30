"""QuantumClassifierModel: Classical neural network with residual and dropout layers."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """
    Classical neural network mirroring a quantum classifier interface.

    The architecture consists of a configurable number of residual blocks,
    each containing two linear layers with ReLU activations and a
    dropout layer.  The final head maps to two output logits.
    """

    def __init__(self, num_features: int, depth: int, dropout: float = 0.2) -> None:
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.network = self._build_network()

    def _build_network(self) -> nn.Module:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            linear1 = nn.Linear(in_dim, self.num_features)
            linear2 = nn.Linear(self.num_features, self.num_features)
            block = nn.Sequential(
                linear1,
                nn.ReLU(),
                nn.Dropout(self.dropout),
                linear2,
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            layers.append(block)
            in_dim = self.num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int, dropout: float = 0.2) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Return a fully‑constructed network along with metadata that is
        analogous to the quantum helper's return signature.
        """
        model = QuantumClassifierModel(num_features, depth, dropout)
        weight_sizes = [p.numel() for p in model.network.parameters()]
        # encoding: indices of input features
        encoding = list(range(num_features))
        # observables: dummy indices representing output classes
        observables = [0, 1]
        return model.network, encoding, weight_sizes, observables

    def __repr__(self) -> str:
        return f"<QuantumClassifierModel depth={self.depth} dropout={self.dropout}>"

__all__ = ["QuantumClassifierModel"]
