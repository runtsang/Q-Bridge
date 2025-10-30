"""Enhanced classical regression model with residual connections and data augmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.05, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.
    The target is a non‑linear function of the input features.
    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the target.
    seed : int | None, optional
        Random seed for reproducibility.
    Returns
    -------
    X, y : np.ndarray
        Feature matrix and target vector.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += rng.normal(scale=noise_std, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Torch dataset returning feature tensors and scalar targets.
    """
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.05, seed: int | None = None):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std, seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ResidualBlock(nn.Module):
    """
    Simple residual block: Linear → ReLU → Linear → addition.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class QModel(nn.Module):
    """
    Classical regression model with residual connections and dropout.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_layer = nn.Linear(num_features, hidden_dim)
        self.residual1 = ResidualBlock(hidden_dim)
        self.residual2 = ResidualBlock(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(state_batch)
        x = torch.relu(x)
        x = self.residual1(x)
        x = torch.relu(x)
        x = self.residual2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.output_layer(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
