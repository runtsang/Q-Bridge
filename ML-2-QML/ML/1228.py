"""Hybrid classical regression model with optional quantum feature support.

This module provides a shared class name `QuantumRegression__gen172` that can be
used in both the classical and quantum branches of the project.  The classical
branch implements a feed‑forward network that can process raw feature vectors
or the output of a quantum circuit.  The data generator supports optional
Gaussian noise on the labels and a reproducible random seed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple, Optional

def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    noise_std: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the target.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    features : np.ndarray of shape (samples, num_features)
        Input feature matrix.
    labels : np.ndarray of shape (samples,)
        Target values.
    """
    rng = np.random.default_rng(random_state)
    features = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = features.sum(axis=1)
    labels = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std is not None:
        labels += rng.normal(scale=noise_std, size=labels.shape)
    return features, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Torch dataset for the synthetic regression data.
    """

    def __init__(
        self,
        samples: int,
        num_features: int,
        *,
        noise_std: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_std=noise_std, random_state=random_state
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression__gen172(nn.Module):
    """
    Classical feed‑forward network that can process raw features or
    the output of a quantum circuit.  The architecture is fully
    configurable via ``hidden_sizes`` and ``dropout``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int,...] = (32, 16),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, input_dim)
            Input features.

        Returns
        -------
        torch.Tensor of shape (batch,)
            Predicted target values.
        """
        return self.net(x).squeeze(-1)

__all__ = ["QuantumRegression__gen172", "RegressionDataset", "generate_superposition_data"]
