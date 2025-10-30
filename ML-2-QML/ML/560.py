"""Classical regression dataset and model mirroring the quantum example with additional extensions.

The model now includes:
- a small linear encoder that projects raw features into a latent space,
- a dropout layer to regularise,
- a lightweight MLP head that outputs a single regression value.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(
    num_features: int,
    samples: int,
    noise: float = 0.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data that mimics the original superposition pattern.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples.
    noise : float, optional
        Standard deviation of Gaussian noise added to the target.
    random_state : int | None, optional
        Seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise > 0.0:
        y += rng.normal(scale=noise, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int, noise: float = 0.0, random_state: int | None = None):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise=noise, random_state=random_state)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    def __init__(self, num_features: int, latent_dim: int = 8, mlp_hidden: int = 32, dropout_p: float = 0.1):
        super().__init__()
        # Encoder projects raw features to latent space
        self.encoder = nn.Sequential(
            nn.Linear(num_features, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        # MLP head maps latent representation to scalar output
        self.head = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encoder(state_batch)
        out = self.head(latent).squeeze(-1)
        return out

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
