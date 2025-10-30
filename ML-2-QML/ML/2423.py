"""Hybrid regression model combining classical neural network with quantum-inspired feature extraction.

This module defines a lightweight classical surrogate that mirrors the structure of the quantum
regression model.  It provides a dataset factory, a dataset class, and a neural network that
accepts both raw features and a quantum‑like embedding.  The network is fully trainable in
PyTorch and can serve as a drop‑in replacement when a quantum backend is unavailable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_hybrid_data(num_features: int, samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset.

    The target is a smooth non‑linear function of the input features, similar to the
    trigonometric form used in the quantum seed.  The function is deliberately
    continuous so that it can be learned by both classical and quantum models.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)


class HybridDataset(Dataset):
    """Dataset that returns raw features and target labels."""

    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = generate_hybrid_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridRegressionModel(nn.Module):
    """Classical neural network that mimics the quantum encoder + variational circuit.

    The architecture consists of:
        * optional quantum‑like feature map (random Fourier features)
        * a shallow feed‑forward network
        * a linear head producing a scalar output
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: list[int] | None = None,
        depth: int = 2,
        use_random_fourier: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        self.use_rf = use_random_fourier

        # Random Fourier feature map that emulates a quantum state preparation
        if self.use_rf:
            self.rf_weights = nn.Parameter(
                torch.randn(num_features, hidden_dims[0]) * np.pi, requires_grad=False
            )
            self.rf_bias = nn.Parameter(
                torch.randn(hidden_dims[0]) * np.pi, requires_grad=False
            )
            in_dim = hidden_dims[0]
        else:
            in_dim = num_features

        layers = []
        current_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ReLU())
            current_dim = h

        # Variational‑style depth
        for _ in range(depth - 1):
            layers.append(nn.Linear(current_dim, current_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_rf:
            # Random Fourier feature map: sin(xW + b)
            x = torch.sin(x @ self.rf_weights + self.rf_bias)
        return self.net(x).squeeze(-1)


__all__ = ["HybridRegressionModel", "HybridDataset", "generate_hybrid_data"]
