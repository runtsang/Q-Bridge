"""Classical classifier factory mirroring the quantum helper interface with extended capabilities."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class ClassicalClassifier(nn.Module):
    """Feed‑forward classifier with optional dropout and configurable depth."""
    def __init__(self, num_features: int, depth: int, num_classes: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_classifier_circuit(num_features: int, depth: int, *,
                             num_classes: int = 2,
                             dropout: float = 0.0) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier and metadata similar to the quantum variant.

    Parameters
    ----------
    num_features:
        Dimensionality of the input vector.
    depth:
        Number of hidden layers.
    num_classes:
        Number of output classes (default 2).
    dropout:
        Dropout probability applied after each hidden layer.

    Returns
    -------
    network:
        ``nn.Sequential`` model.
    encoding:
        Identity mapping of feature indices.
    weight_sizes:
        Number of trainable parameters per linear layer.
    observables:
        Target class indices.
    """
    model = ClassicalClassifier(num_features, depth, num_classes, dropout)
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for m in model.net:
        if isinstance(m, nn.Linear):
            weight_sizes.append(m.weight.numel() + m.bias.numel())
    observables = list(range(num_classes))
    return model, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
