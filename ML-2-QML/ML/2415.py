"""Hybrid classical classifier that can consume quantum‑generated features."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridClassifier(nn.Module):
    """Feed‑forward classifier that accepts either raw features or quantum‑encoded features.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature space.
    depth : int
        Number of hidden layers in the network.
    """
    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return a Classical feed‑forward network along with metadata."""
    network = HybridClassifier(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in network.parameters()]
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class SamplerQNN(nn.Module):
    """Simple classical sampler network used to mimic quantum sampling."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


__all__ = ["HybridClassifier", "build_classifier_circuit", "SamplerQNN"]
