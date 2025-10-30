"""Classical classifier module that mirrors the quantum interface and adds regression‑derived data augmentation.

Features
--------
* Deep ReLU network with optional residual connections.
* Dropout and batch‑norm for regularisation.
* Data generation via the superposition data routine from the regression seed, thresholded to produce binary labels.
* Utility to expose weight sizes for debugging and scaling analysis.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and a target function.

    The labels are thresholded to produce a binary classification task.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    # Binary labels: 1 if y > 0 else 0
    labels = (y > 0).astype(np.float32)
    return x, labels


class ClassificationDataset(torch.utils.data.Dataset):
    """Dataset that wraps the superposition data for binary classification."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumClassifierModel(nn.Module):
    """Depth‑aware feed‑forward classifier with optional residual blocks."""

    def __init__(self, num_features: int, hidden_dim: int = 64, depth: int = 4,
                 dropout: float = 0.1, use_residual: bool = False):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.extend([linear, nn.ReLU(), nn.BatchNorm1d(hidden_dim), self.dropout])
            in_dim = hidden_dim
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch_size, 2)."""
        x = state_batch.to(torch.float32)
        if self.use_residual:
            # Simple residual skip connection from input to first hidden layer
            residual = x
            x = self.body(x)
            x = x + residual[:, :self.hidden_dim]
        else:
            x = self.body(x)
        logits = self.head(x)
        return logits

    @staticmethod
    def weight_sizes(model: nn.Module) -> List[int]:
        """Return a list of weight+bias counts for each trainable layer."""
        sizes = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                sizes.append(module.weight.numel() + module.bias.numel())
        return sizes

    @staticmethod
    def observables() -> Iterable[int]:
        """Placeholder for compatibility with the quantum interface."""
        return [0, 1]


__all__ = ["QuantumClassifierModel", "ClassificationDataset", "generate_superposition_data"]
