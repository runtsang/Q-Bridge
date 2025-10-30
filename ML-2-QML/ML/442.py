"""Classical classifier that mirrors the quantum helper interface and adds advanced training utilities."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dim: int = 64,
    dropout: float = 0.1,
) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Construct a multi‑layer perceptron with optional dropout.

    Returns:
        network: nn.Sequential model
        encoding: list of feature indices (identity)
        weight_sizes: list of trainable parameter counts per layer
        observables: list of output node indices (here just [0,1] for binary logits)
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding: List[int] = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, hidden_dim)
        layers.append(linear)
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = hidden_dim

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)

    observables = [0, 1]
    return network, encoding, weight_sizes, observables


class QuantumClassifierModel(nn.Module):
    """Wrapper around the classical network exposing the same interface as the quantum helper."""
    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth, hidden_dim, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(F.softmax(logits, dim=-1), dim=-1)

    def get_weight_sizes(self) -> List[int]:
        return self.weight_sizes


__all__ = ["build_classifier_circuit", "QuantumClassifierModel"]
