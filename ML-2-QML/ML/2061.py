"""Hybrid classical regression with configurable dropout and batch‑norm.

The module extends the original seed by adding a flexible hidden‑layer
architecture, optional dropout, and a noise‑aware data generator.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional, Tuple


def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    noise: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics a quantum superposition.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of data points.
    noise : float, default 0.05
        Standard deviation of Gaussian noise added to the labels.
    rng : np.random.Generator, optional
        Random number generator, for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Features array of shape (samples, num_features) and labels array of
        shape (samples,).
    """
    rng = rng or np.random.default_rng(42)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += rng.normal(scale=noise, size=y.shape).astype(np.float32)
    return x, y


class RegressionDataset(Dataset):
    """
    Torch dataset wrapping the synthetic superposition data.

    The `__getitem__` method returns a dictionary with keys ``states`` and
    ``target`` to stay compatible with the original API.
    """

    def __init__(self, samples: int, num_features: int, *, noise: float = 0.05):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise=noise
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """
    A dense neural network that optionally uses dropout and batch‑norm.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    hidden_dims : Tuple[int,...], optional
        Sizes of hidden layers. Defaults to ``(64, 32)``.
    dropout : float, optional
        Dropout probability. ``0.0`` disables dropout.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: Tuple[int,...] = (64, 32),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = num_features
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
