"""HybridClassifier: classical neural network with optional quantum interface.

This module implements a fully‑connected network that can be used as a drop‑in
replacement for the original `build_classifier_circuit`.  It exposes the same
metadata interface (`encoding`, `weights`, `observables`) so that downstream
experiments can interoperate with the quantum counterpart.

Key extensions:
- Support for arbitrary depth and hidden layer size.
- Optional dropout and batch‑norm for regularisation.
- Parameter extraction method that returns a tuple compatible with the Qiskit
  implementation (`encoding`, `weights`, `observables`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

class HybridClassifier(nn.Module):
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 32,
                 depth: int = 3,
                 dropout: float = 0.0,
                 use_batchnorm: bool = False) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features

        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(linear)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.classifier(h)

    def metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return encoding indices, weight counts and observable indices.

        The convention matches the Qiskit implementation: `encoding` are the
        indices of the input features, `weights` are the total number of
        trainable parameters, and `observables` are dummy indices for the two
        output logits.
        """
        encoding = list(range(self.features[0].in_features))
        weight_sizes = []
        for m in self.features:
            if isinstance(m, nn.Linear):
                weight_sizes.append(m.weight.numel() + m.bias.numel())
        weight_sizes.append(self.classifier.weight.numel() + self.classifier.bias.numel())
        observables = list(range(2))
        return encoding, weight_sizes, observables

__all__ = ["HybridClassifier"]
