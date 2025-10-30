"""Enhanced classical regression model with richer data and residual connections."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple


def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.05,
    mix_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset where each sample is a mixture of a clean
    superposition‑derived signal and a Gaussian‑noise‑corrupted version.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of samples to generate.
    noise_std : float, default 0.05
        Standard deviation of the Gaussian noise added to the signal.
    mix_ratio : float, default 0.5
        Weighting between the clean signal (1 - mix_ratio) and the noisy
        perturbation (mix_ratio).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``X`` of shape (samples, num_features) and ``y`` of shape (samples,).
    """
    # Clean signal: sinusoidal superposition as in the seed
    x_clean = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x_clean.sum(axis=1)
    y_clean = np.sin(angles) + 0.1 * np.cos(2 * angles)

    # Gaussian noise component
    noise = np.random.normal(scale=noise_std, size=samples).astype(np.float32)

    # Mix clean and noisy signals
    y = (1 - mix_ratio) * y_clean + mix_ratio * noise
    return x_clean, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset that returns a dictionary with features and target.
    """

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualDenseBlock(nn.Module):
    """
    A small residual block consisting of two linear layers with a ReLU
    activation and a skip connection.
    """

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.relu(out + residual)


class QModel(nn.Module):
    """
    Classical regression model that uses a residual‑dense block before
    the final linear output layer.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.initial = nn.Linear(num_features, 64)
        self.res_block = ResidualDenseBlock(64, 128)
        self.final = nn.Linear(64, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.initial(state_batch)
        x = self.res_block(x)
        return self.final(x).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
