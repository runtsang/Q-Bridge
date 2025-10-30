"""Hybrid ML model combining classical feed-forward and quantum-inspired design.

This module provides a reusable classifier/regressor factory that
mirrors the quantum helper interface.  The returned metadata is
compatible with the quantum side so that classical and quantum
experiments can be swapped seamlessly.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Construct a feed‑forward classifier with depth‑controlled hidden layers.

    Returns:
        network: nn.Sequential classifier
        encoding: list of feature indices used as “encoding” (identity mapping)
        weight_sizes: cumulative number of trainable parameters per layer
        observables: placeholder list for compatibility with quantum circuit
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU(inplace=True)])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

def build_regression_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Construct a feed‑forward regressor with depth‑controlled hidden layers.

    Returns a network that outputs a single scalar.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU(inplace=True)])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 1)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0]  # dummy for compatibility
    return network, encoding, weight_sizes, observables

def generate_classification_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary labels from a simple non‑linear decision boundary."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    y = (np.sum(x, axis=1) > 0).astype(np.int64)
    return x, y

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classification_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.long),
        }

class RegressionDataset(torch.utils.data.Dataset):
    """Same as in the regression seed but re‑exported for consistency."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the same superposition dataset used in the quantum regression seed."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class HybridModel(nn.Module):
    """A simple two‑head model that can be used for classification or regression.

    The network is a shared trunk followed by a task‑specific head.
    """
    def __init__(self, num_features: int, depth: int, task: str = "classification"):
        super().__init__()
        trunk = []
        in_dim = num_features
        for _ in range(depth):
            trunk.append(nn.Linear(in_dim, num_features))
            trunk.append(nn.ReLU(inplace=True))
            in_dim = num_features
        self.trunk = nn.Sequential(*trunk)

        if task == "classification":
            self.head = nn.Linear(in_dim, 2)
        else:
            self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        return self.head(x).squeeze(-1)

__all__ = [
    "build_classifier_circuit",
    "build_regression_circuit",
    "generate_classification_data",
    "generate_superposition_data",
    "ClassificationDataset",
    "RegressionDataset",
    "HybridModel",
]
