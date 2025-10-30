"""Extended classical regression model with advanced preprocessing and a deeper neural network.

This module builds on the original seed by adding:
* A Fourier feature mapping to capture periodic structure.
* Residual blocks with batch‑norm and dropout for regularisation.
* A configurable number of layers and hidden units.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_fourier_features(features: np.ndarray, n_fourier: int = 8) -> np.ndarray:
    """Map real features to a higher‑dimensional Fourier feature space."""
    x = features.astype(np.float32)
    # random frequencies
    freqs = np.random.uniform(0.5, 1.5, size=(x.shape[1], n_fourier))
    fourier = np.concatenate([np.sin(x @ freqs), np.cos(x @ freqs)], axis=1)
    return fourier.astype(np.float32)


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression target from a sinusoidal function with noise."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset providing Fourier‑mapped features and regression targets."""
    def __init__(self, samples: int, num_features: int, n_fourier: int = 8):
        raw_x, raw_y = generate_superposition_data(num_features, samples)
        self.features = generate_fourier_features(raw_x, n_fourier)
        self.labels = raw_y

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """A residual block consisting of two linear layers with batch‑norm and dropout."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RegressionModel(nn.Module):
    """Deep residual network for regression."""
    def __init__(self, num_features: int, n_fourier: int = 8, hidden_dim: int = 64, depth: int = 4):
        super().__init__()
        input_dim = num_features * n_fourier * 2  # sin + cos
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.input_layer(state_batch)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x).squeeze(-1)


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
