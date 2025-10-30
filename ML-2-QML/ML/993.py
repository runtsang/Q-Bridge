"""Extended classical regression model with configurable architecture and data scaling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data mimicking a superposition of two basis states.
    The target is a smooth non‑linear function of the feature sum.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns feature tensors and scalar targets."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """
    Configurable feed‑forward network for regression.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    hidden_dims : list[int], optional
        Sizes of hidden layers. Defaults to [32, 16].
    activation : nn.Module, optional
        Activation function to use. Defaults to nn.ReLU().
    dropout : float, optional
        Dropout probability. Defaults to 0.0 (no dropout).
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: list[int] | None = None,
        activation: nn.Module | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]
        activation = activation or nn.ReLU()
        layers = []
        in_dim = num_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
