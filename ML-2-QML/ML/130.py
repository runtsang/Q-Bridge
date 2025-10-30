"""Enhanced classical regression module with residual blocks and multi‑head output.

The module mirrors the original `QModel` but adds:
* `StandardScaler` for input normalisation.
* A residual block and dropout for better regularisation.
* A dual‑head that predicts both the mean target and its variance.
* A simple `train` helper that runs a single optimisation step.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic superposition‑style regression target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns normalised state vectors and scalar targets."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """A simple residual block with dropout."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


class QuantumRegression(nn.Module):
    """Classical regression model with residuals and a variance head."""

    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
        )
        # Dual heads: mean and log‑variance
        self.mean_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(state_batch)
        mean = self.mean_head(features).squeeze(-1)
        logvar = self.logvar_head(features).squeeze(-1)
        return mean, logvar

    def loss(self, mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative log‑likelihood for Gaussian output."""
        var = torch.exp(logvar)
        return 0.5 * torch.mean((target - mean) ** 2 / var + logvar)

    def train_step(self, optimizer: torch.optim.Optimizer, batch: dict):
        self.train()
        optimizer.zero_grad()
        mean, logvar = self.forward(batch["states"])
        loss = self.loss(mean, logvar, batch["target"])
        loss.backward()
        optimizer.step()
        return loss.item()
