"""Hybrid classical classifier that mirrors the quantum interface and incorporates regression-inspired depth.

The class exposes a build_classifier_circuit function that returns a neural network, an encoding list,
weight sizes, and observable indices, matching the signature used by the quantum counterpart.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate superposition feature vectors and binary labels derived from a sinusoidal function."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = (np.sin(angles) > 0).astype(np.float32)
    return x, y


class ClassificationDataset(Dataset):
    """Dataset yielding superposition feature vectors and binary labels."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridClassifier(nn.Module):
    """Classical classifier that mimics the quantum variational ansatz."""

    def __init__(self, num_features: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Return a classical network, encoding indices, weight sizes, and observable indices."""
    net = HybridClassifier(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for module in net.modules():
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())
    observables = [0, 1]
    return net, encoding, weight_sizes, observables
