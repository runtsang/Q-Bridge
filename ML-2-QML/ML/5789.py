"""Hybrid classical classifier that mirrors the quantum helper interface with weight sharing."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """A feed‑forward network with a shared linear layer across all hidden layers."""

    def __init__(self, num_features: int, depth: int, hidden_dim: int | None = None):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features

        # Shared weight and bias for all hidden layers
        self.shared_weight = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        self.shared_bias = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.xavier_uniform_(self.shared_weight)
        nn.init.zeros_(self.shared_bias)

        # Final classification head
        self.head = nn.Linear(self.hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared linear transformation, ReLU, and the head."""
        out = F.linear(x, self.shared_weight, self.shared_bias)
        for _ in range(self.depth):
            out = F.relu(F.linear(out, self.shared_weight, self.shared_bias))
        out = self.head(out)
        return out

    def get_weight_sizes(self) -> List[int]:
        """Return the number of trainable parameters in each component."""
        return [
            self.shared_weight.numel() + self.shared_bias.numel(),
            self.head.weight.numel() + self.head.bias.numel(),
        ]


def build_classifier_circuit(
    num_features: int, depth: int, hidden_dim: int | None = None
) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """Factory that reproduces the seed API while using the new model."""
    network = QuantumClassifierModel(num_features, depth, hidden_dim)
    encoding = list(range(num_features))
    weight_sizes = network.get_weight_sizes()
    observables = list(range(2))
    return network, encoding, weight_sizes, observables
