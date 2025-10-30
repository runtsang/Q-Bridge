"""Hybrid classical classifier with enhanced regularization and modular metadata."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier and return metadata.

    The network now includes batch‑norm after each hidden layer and a final dropout
    layer controlled by the caller.  The returned ``weight_sizes`` list can be used
    for fine‑grained weight‑decay schedules.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.BatchNorm1d(num_features), nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class HybridClassifier(nn.Module):
    """
    A torch ``nn.Module`` that wraps the classic feed‑forward circuit with
    optional dropout and L2 regularization.  The class exposes the same
    ``build_classifier_circuit`` interface used by the quantum counterpart,
    making it drop‑in compatible in hybrid pipelines.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        dropout: float = 0.0,
        l2: float = 0.0,
    ) -> None:
        super().__init__()
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth
        )
        if dropout > 0.0:
            self.network.add_module("dropout", nn.Dropout(dropout))
        self.l2 = l2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    @property
    def num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def weight_decay_params(self) -> Iterable[nn.Parameter]:
        """Yield parameters that should receive L2 regularization."""
        if self.l2 == 0.0:
            return iter(())
        return iter(self.parameters())


__all__ = ["HybridClassifier", "build_classifier_circuit"]
