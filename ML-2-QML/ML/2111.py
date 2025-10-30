"""Enhanced classical regression model inspired by quantum data generation.

The module mirrors the original interface but introduces a trigonometric
feature mapper and a deeper neural network.  The mapper transforms the raw
features into a richer basis that is reminiscent of quantum state
amplitudes, while the network learns regression in this augmented space.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics quantum superposition.

    Parameters
    ----------
    num_features : int
        Number of classical input dimensions.
    samples : int
        Number of samples to generate.

    Returns
    -------
    features, labels : tuple[np.ndarray, np.ndarray]
        ``features`` has shape ``(samples, num_features)`` and is drawn
        uniformly from ``[-1, 1]``. ``labels`` are a nonlinear function
        of the input that contains both sine and cosine components.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns a dictionary with ``states`` and ``target``."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class TrigFeatureMapper(nn.Module):
    """Map real features to a sine/cosine basis."""

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, num_features)
        sin_part = torch.sin(x)
        cos_part = torch.cos(x)
        # Concatenate along feature axis
        return torch.cat([sin_part, cos_part], dim=-1)


class QModel(nn.Module):
    """
    Classical regression model that first applies a trigonometric feature
    mapper and then passes the result through a moderately sized MLP.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.mapper = TrigFeatureMapper(num_features)
        # After mapping we have 2 * num_features features
        hidden_dim = max(64, 2 * num_features)
        self.net = nn.Sequential(
            nn.Linear(2 * num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        mapped = self.mapper(state_batch)
        return self.net(mapped).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
