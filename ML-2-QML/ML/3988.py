"""Hybrid regression model – classical implementation.

The class shares the name with the quantum counterpart but remains
entirely classical: a random linear encoder (mimicking a quantum
encoding) followed by a depth‑controlled multilayer perceptron.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression targets from a superposition-like
    construction.  Each sample is a ``num_features``‑dimensional vector
    uniformly drawn from [−1, 1] and the target is a smooth sine‑cosine
    combination of the sum of elements.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns raw feature vectors and targets."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RandomLinearEncoder(nn.Module):
    """Fixed random linear transformation used as a classical encoder."""

    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.register_buffer(
            "weight",
            torch.tensor(
                rng.standard_normal((out_features, in_features), dtype=np.float32),
                dtype=torch.float32,
            ),
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class HybridRegressionModel(nn.Module):
    """
    Classical hybrid model: a fixed random encoder followed by a
    depth‑controlled feed‑forward network.  The architecture mirrors the
    quantum version’s depth and feature count but remains entirely
    trainable with standard back‑propagation.
    """

    def __init__(self, num_features: int, depth: int = 2, hidden_dim: int | None = None):
        """
        Parameters
        ----------
        num_features:
            Dimensionality of the input feature vectors.
        depth:
            Number of hidden layers in the MLP.  A depth of 0 yields a
            single linear head.
        hidden_dim:
            Width of hidden layers.  If ``None`` defaults to ``num_features``.
        """
        super().__init__()
        hidden_dim = hidden_dim or num_features
        self.encoder = RandomLinearEncoder(num_features, hidden_dim)
        layers = [self.encoder]
        in_dim = hidden_dim

        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.extend([linear, nn.ReLU()])
            in_dim = hidden_dim

        head = nn.Linear(in_dim, 1)
        layers.append(head)

        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        states:
            Tensor of shape ``(batch, num_features)``.
        """
        return self.net(states).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
