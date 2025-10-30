"""Hybrid classical model mirroring the quantum interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import numpy as np


def generate_classification_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic binary classification data with a non‑linear decision boundary."""
    X = np.random.randn(samples, num_features).astype(np.float32)
    y = (np.sum(X ** 2, axis=1) > num_features).astype(np.int64)
    return X, y


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Superposition data for the regression task."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class ClassificationDataset(torch.utils.data.Dataset):
    """Dataset wrapper for the synthetic classification data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classification_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper for the superposition regression data."""

    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def build_hybrid_circuit(num_qubits: int, depth: int, task: str = "classification") -> Tuple[Iterable[int], Iterable[int], List[int]]:
    """
    Return metadata that mirrors the quantum circuit:
    * encoding indices
    * weight indices
    * observable indices
    """
    encoding = list(range(num_qubits))
    weight_sizes = [num_qubits] * depth
    observables = list(range(num_qubits))
    return encoding, weight_sizes, observables


class HybridClassifier(nn.Module):
    """
    Classical neural network that mirrors the interface of the quantum hybrid model.
    The network can be configured for classification or regression.
    """

    def __init__(
        self,
        num_features: int,
        num_qubits: int,
        depth: int,
        task: str = "classification",
        hidden_dims: Tuple[int,...] = (32, 16),
    ):
        super().__init__()
        self.task = task
        self.encoder = nn.Linear(num_features, num_qubits, bias=False)
        layers: List[nn.Module] = []
        in_dim = num_qubits
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        if task == "classification":
            layers.append(nn.Linear(in_dim, 2))
            self.head = nn.Sequential(*layers, nn.LogSoftmax(dim=-1))
        else:  # regression
            layers.append(nn.Linear(in_dim, 1))
            self.head = nn.Sequential(*layers)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that applies an encoding and then the classical head."""
        encoded = self.encoder(x)
        return self.head(encoded)


__all__ = [
    "HybridClassifier",
    "ClassificationDataset",
    "RegressionDataset",
    "build_hybrid_circuit",
    "generate_classification_data",
    "generate_superposition_data",
]
