"""Hybrid classical model combining classification and regression capabilities.

This module extends the original `QuantumClassifierModel` by adding a regression
head and by exposing a multi‑task dataset that can be used for both
classification and regression.  The network trunk is shared, which
reduces memory footprint and allows joint training if desired.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data in the form of a superposition of |0...0> and |1...1>.

    The labels are a continuous target derived from sin(2·θ)·cos(φ).  For
    classification the same data is used but the target is thresholded at
    zero to produce a binary label.
    """
    omega_0 = np.zeros(2 ** num_features, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_features, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_features), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class SuperpositionDataset(Dataset):
    """Dataset that yields either a regression target or a binary classification label."""

    def __init__(self, samples: int, num_features: int, task: str = "regression"):
        self.states, self.labels = generate_superposition_data(num_features, samples)
        if task == "classification":
            # Convert continuous labels into binary labels
            self.labels = (self.labels > 0).astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumClassifierModel(nn.Module):
    """Shared trunk with two heads: classification (2‑way) and regression (scalar)."""

    def __init__(self, num_features: int, hidden_sizes: Tuple[int,...] = (32, 16), task: str = "classification"):
        super().__init__()
        layers = []
        in_dim = num_features
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            in_dim = size
        self.trunk = nn.Sequential(*layers)

        self.classification_head = nn.Linear(in_dim, 2)
        self.regression_head = nn.Linear(in_dim, 1)

        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        trunk_out = self.trunk(x)
        if self.task == "classification":
            return self.classification_head(trunk_out)
        return self.regression_head(trunk_out).squeeze(-1)

    def switch_task(self, task: str) -> None:
        """Change the active head."""
        assert task in {"classification", "regression"}
        self.task = task

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """Return a feed‑forward network mirroring the quantum interface.

        The returned network contains a shared trunk and a classification head.
        The metadata lists the indices of the encoding parameters and the
        sizes of all learnable weight tensors.
        """
        layers = []
        in_dim = num_features
        weight_sizes = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        encoding = list(range(num_features))
        observables = list(range(2))
        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel", "SuperpositionDataset", "generate_superposition_data"]
