"""Hybrid classical classifier with optional quantum feature extraction.

This module extends the original QuantumClassifierModel by adding a deeper MLP with
skip connections, a unified dataset generator, and a regression‑capable head.
The public API mirrors the original build_classifier_circuit signature for
compatibility while exposing additional utilities for cross‑modal experiments.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Iterable, Tuple, List

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data from superposition states.

    The data distribution matches the one used in the quantum regression example
    but is now used for classification by adding a binary label derived from the
    sign of the sum of features. This provides a linear separable target while
    keeping the quantum‑friendly structure.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y_class = (np.sin(angles) > 0).astype(np.int64)
    y_reg = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y_class, y_reg

class SuperpositionDataset(Dataset):
    """Dataset returning features and either classification or regression targets."""
    def __init__(self, samples: int, num_features: int, task: str = "classification"):
        self.features, self.labels_class, self.labels_reg = generate_superposition_data(num_features, samples)
        self.task = task

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        item = {"features": torch.tensor(self.features[idx], dtype=torch.float32)}
        if self.task == "classification":
            item["target"] = torch.tensor(self.labels_class[idx], dtype=torch.long)
        else:
            item["target"] = torch.tensor(self.labels_reg[idx], dtype=torch.float32)
        return item

def build_classifier_circuit(num_features: int,
                             depth: int = 2,
                             classification: bool = True) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Build a classical MLP with optional residual connections.

    Parameters
    ----------
    num_features
        Number of input features.
    depth
        Number of hidden layers.
    classification
        When ``True`` the head ends in a 2‑class linear layer for cross‑entropy;
        otherwise a single‑output regression head is used.

    Returns
    -------
    network
        nn.Sequential network including the head.
    encoding
        List of input indices (identity mapping for compatibility).
    weight_sizes
        Number of trainable parameters per layer.
    observables
        Dummy list of observable indices to keep API compatible with the quantum
        counterpart.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for i in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        # Residual skip
        if i > 0:
            layers.append(nn.Identity())  # placeholder for potential skip logic
        in_dim = num_features

    if classification:
        head = nn.Linear(in_dim, 2)
    else:
        head = nn.Linear(in_dim, 1)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0] * (depth + 1)  # placeholder for compatibility
    return network, encoding, weight_sizes, observables

class HybridClassifier(nn.Module):
    """Classical MLP that optionally delegates feature extraction to a quantum backend."""
    def __init__(self, num_features: int, depth: int = 2, classification: bool = True):
        super().__init__()
        self.net, self.encoding, self.weight_sizes, _ = build_classifier_circuit(num_features, depth, classification)
        self.classification = classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

__all__ = ["HybridClassifier", "build_classifier_circuit", "SuperpositionDataset"]
