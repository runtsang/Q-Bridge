"""Extended classical regression model with residual blocks and dropout.

The module mirrors the original ``QModel`` and ``RegressionDataset`` but
adds additional layers, residual connections and dropout for better
generalisation.  The public API remains unchanged so existing scripts
continue to work unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data inspired by a quantum superposition.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.

    Returns
    -------
    X : np.ndarray of shape (samples, num_features)
        Uniformly distributed features in ``[-1, 1]``.
    y : np.ndarray of shape (samples,)
        Target values computed as ``sin(sum(x)) + 0.1*cos(2*sum(x))``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic superposition data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """A simple residual block with two linear layers and a ReLU."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        return self.activation(out)


class QModel(nn.Module):
    """
    Classical regression network with configurable depth.

    The network consists of an initial linear layer, followed by a
    sequence of residual blocks, a dropout layer and a final linear
    head that outputs a single regression value.
    """

    def __init__(self, num_features: int, hidden_dim: int = 32,
                 num_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.initial = nn.Linear(num_features, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.final_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.initial(state_batch)
        for block in self.blocks:
            x = block(x)
        x = self.final_dropout(x)
        return self.head(x).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
