"""UnifiedClassifier: Classical implementation.

This module defines a class that builds a feature‑engineered feed‑forward
network.  The architecture mirrors the quantum interface: the first
layer applies an RBF feature map, subsequent layers are linear + ReLU
and the final head produces two logits.  The method ``build_model`` returns
the network together with metadata (encoding indices, weight sizes,
observables) so that it can be swapped with the quantum version.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import numpy as np


class RBFFeature(nn.Module):
    """Simple RBF feature map applied element‑wise."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * x * x)


class UnifiedClassifier(nn.Module):
    """Classical classifier with RBF feature map and depth‑controlled MLP."""

    def __init__(self, num_features: int, depth: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.gamma = gamma

        # RBF feature extractor
        self.rbf = RBFFeature(gamma)

        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        # First linear layer after RBF
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

        # Hidden layers
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        # Output head
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        self.network = nn.Sequential(*layers)
        self.observables = list(range(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rbf(x)
        return self.network(x)

    def build_model(self) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Return the network and metadata compatible with the quantum API."""
        encoding = list(range(self.num_features))
        weight_sizes = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                weight_sizes.append(module.weight.numel() + module.bias.numel())
        observables = self.observables
        return self.network, encoding, weight_sizes, observables


__all__ = ["UnifiedClassifier"]
