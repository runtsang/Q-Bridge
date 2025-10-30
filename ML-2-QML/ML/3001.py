"""Hybrid classifier combining classical neural net with quantum feature extractor."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class HybridClassifier(nn.Module):
    """Classic neural network that maps quantum measurement results to class logits."""
    def __init__(self, num_features: int, depth: int, hidden_sizes: List[int] | None = None) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [num_features, num_features]
        layers: List[nn.Module] = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Return a classical classifier and metadata mirroring the quantum interface.
    The function signature matches the quantum version so that the two halves
    can be swapped or trained jointly.
    """
    classifier = HybridClassifier(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in classifier.parameters()]
    observables = list(range(2))
    return classifier, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
# Regression submodule – identical to the EstimatorQNN example
# --------------------------------------------------------------------------- #
class EstimatorNN(nn.Module):
    """Small regression network used by the quantum estimator."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def EstimatorQNN() -> EstimatorNN:
    """Return an instance of the regression network."""
    return EstimatorNN()


__all__ = ["HybridClassifier", "build_classifier_circuit", "EstimatorNN", "EstimatorQNN"]
