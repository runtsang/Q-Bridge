"""Enhanced classical regression model with residual connections and dropout."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data from a superposition of basis states.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature space.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Features of shape (samples, num_features) and labels of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += np.random.normal(scale=noise_std, size=y.shape)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the synthetic regression data.
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


class ResidualBlock(nn.Module):
    """
    Simple residual block with batch normalization and dropout.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + x)


class QModel(nn.Module):
    """
    Residual neural network for regression with dropout and batch normalization.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.input_layer = nn.Linear(num_features, 64)
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(state_batch)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.dropout(x)
        return self.output_layer(x).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
