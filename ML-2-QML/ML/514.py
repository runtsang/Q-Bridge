"""Hybrid classical regression model with configurable depth and regularisation.

This module extends the original seed by allowing arbitrary hidden layer
configurations, dropout, and optional noise injection during data generation.
The public API is intentionally identical to the quantum counterpart so that
experiments can be swapped by changing the import path.

Author: GPT-OSS-20B
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_level: float = 0.0,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where the target is a smooth
    function of the sum of the input features.

    Parameters
    ----------
    num_features : int
        Dimensionality of each sample.
    samples : int
        Number of samples to generate.
    noise_level : float, default 0.0
        Standard deviation of Gaussian noise added to the target.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    x : np.ndarray of shape (samples, num_features)
        Feature matrix.
    y : np.ndarray of shape (samples,)
        Target vector.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_level > 0.0:
        y += rng.normal(scale=noise_level, size=y.shape)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the synthetic regression data.

    The dataset returns a dictionary containing the feature vector
    (under key ``states``) and the scalar target (under key ``target``).
    """

    def __init__(self, samples: int, num_features: int, noise_level: float = 0.0, *, seed: int | None = None):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_level=noise_level, seed=seed
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegression(nn.Module):
    """
    Fully‑connected regression network with flexible architecture.

    Parameters
    ----------
    num_features : int
        Size of the input feature vector.
    hidden_sizes : Sequence[int], optional
        List of hidden layer sizes. Defaults to ``[32, 16]``.
    dropout : float, optional
        Dropout probability applied after each hidden layer. Defaults to ``0.0``.
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: list[int] | tuple[int,...] = (32, 16),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor of shape (batch, num_features)

        Returns
        -------
        torch.Tensor of shape (batch,)
            Predicted scalar values.
        """
        return self.net(state_batch).squeeze(-1)


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
