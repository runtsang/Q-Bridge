"""Enhanced classical regression model with residual blocks, dropout, and stochastic data augmentation.

This module extends the original `QuantumRegression` seed by adding:
- A residual neural network backbone with configurable hidden layers.
- Optional dropout and batch‑normalization for regularisation.
- Data augmentation that randomly shifts the input features during training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]

def generate_superposition_data(
    num_features: int,
    samples: int,
    augmentation: bool = False,
    noise_std: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that follows a superposition‑like pattern.
    Parameters
    ----------
    num_features : int
        Dimensionality of the input space.
    samples : int
        Number of samples to generate.
    augmentation : bool, optional
        If ``True`` add a small random shift to each feature vector.
    noise_std : float, optional
        Standard deviation of the Gaussian noise added to the target when
        ``augmentation`` is enabled.
    Returns
    -------
    x : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Target vector of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)

    if augmentation:
        shift = np.random.uniform(-0.1, 0.1, size=x.shape).astype(np.float32)
        x += shift
        y += np.random.normal(scale=noise_std, size=y.shape).astype(np.float32)

    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary with keys ``states`` and ``target``.
    The ``states`` tensor contains the feature vector and ``target`` holds the
    corresponding regression label.
    """
    def __init__(self, samples: int, num_features: int, augmentation: bool = False):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, augmentation=augmentation
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ResidualBlock(nn.Module):
    """
    A simple residual block: Linear → ReLU → Linear → add input.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # If dimensions differ, adjust the shortcut
        if in_features!= out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += identity
        return self.relu(out)

class QModel(nn.Module):
    """
    Residual neural network for regression.
    Parameters
    ----------
    num_features : int
        Input dimensionality.
    hidden_sizes : list[int], optional
        Sequence of hidden layer sizes. Defaults to [32, 16].
    dropout : float, optional
        Dropout probability applied after each residual block.
    """
    def __init__(
        self,
        num_features: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [32, 16]
        layers = []
        in_dim = num_features
        for out_dim in hidden_sizes:
            layers.append(ResidualBlock(in_dim, out_dim, dropout=dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)
