"""Classical hybrid classifier with optional regression head.

This module extends the original `build_classifier_circuit` to expose
weight‑size metadata and adds a lightweight regression head.
The interface mirrors the quantum counterpart so that the same
high‑level API can be used in a purely classical setting.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a feed‑forward classifier and metadata similar to the quantum variant."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


def build_regressor_circuit(num_features: int, hidden: int = 32) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a simple regression network with a single output."""
    layers: List[nn.Module] = [
        nn.Linear(num_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    ]
    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    weight_sizes = [
        layer.weight.numel() + layer.bias.numel()
        for layer in network
        if isinstance(layer, nn.Linear)
    ]
    observables = [0]
    return network, encoding, weight_sizes, observables


class QuantumClassifierModel(nn.Module):
    """Classical model that can perform classification or regression.

    The class exposes the same public API as the quantum implementation
    (`forward`, `build_*_circuit`, `get_weight_sizes`) so that downstream
    code can switch between classical and quantum backends without
    modification.
    """

    def __init__(self, num_features: int, depth: int = 3, regression: bool = False):
        super().__init__()
        self.classifier, _, self.class_weights, _ = build_classifier_circuit(num_features, depth)
        self.regression = regression
        if regression:
            self.regressor, _, self.reg_weights, _ = build_regressor_circuit(num_features)
        else:
            self.regressor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)
        if self.regression and self.regressor is not None:
            reg = self.regressor(x)
            return logits, reg
        return logits

    def get_weight_sizes(self) -> Tuple[List[int], List[int]]:
        """Return weight sizes for classifier and regressor heads."""
        if self.regression:
            return self.class_weights, self.reg_weights
        return self.class_weights, []


__all__ = ["QuantumClassifierModel", "build_classifier_circuit", "build_regressor_circuit"]
