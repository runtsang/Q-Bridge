"""Hybrid classical neural network that mirrors the quantum classifier interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Build a deep feed‑forward network with a fixed encoding scheme and return metadata.
    The function is intentionally compatible with the quantum helper, exposing the same
    `encoding`, `weight_sizes` and `observables` placeholders.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding: List[int] = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables: List[int] = [0, 1]  # placeholder for class labels
    return network, encoding, weight_sizes, observables


class HybridClassifier(nn.Module):
    """
    Classical feed‑forward classifier that shares the interface of the quantum
    counterpart.  It can optionally attach a regression head, enabling a
    unified training loop with a quantum estimator.
    """

    def __init__(self, num_features: int, depth: int, num_classes: int = 2, regression: bool = False) -> None:
        super().__init__()
        self.encoder, self.encoding, self.weight_sizes, _ = build_classifier_circuit(num_features, depth)
        self.classifier = nn.Linear(num_features, num_classes)
        self.weight_sizes.append(self.classifier.weight.numel() + self.classifier.bias.numel())

        self.regression = regression
        if regression:
            self.regressor = nn.Linear(num_features, 1)
            self.weight_sizes.append(self.regressor.weight.numel() + self.regressor.bias.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = self.encoder(x)
        out = self.classifier(h)
        if self.regression:
            reg = self.regressor(h)
            return out, reg
        return out

    def parameters_list(self) -> List[torch.nn.Parameter]:
        """Return all parameters in the same order as the quantum circuit."""
        params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        if self.regression:
            params += list(self.regressor.parameters())
        return params


def EstimatorQNN() -> nn.Module:
    """
    Simple regression network inspired by the qiskit EstimatorQNN example.
    """
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()


__all__ = ["HybridClassifier", "EstimatorQNN", "build_classifier_circuit"]
