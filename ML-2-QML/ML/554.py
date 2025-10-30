"""Classical regression dataset and model with enhanced training utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split


def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.05,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 1‑D superposition data with optional Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += rng.normal(scale=noise_std, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Wrapper around the synthetic dataset for PyTorch."""

    def __init__(
        self,
        samples: int,
        num_features: int,
        noise_std: float = 0.05,
        seed: int | None = None,
    ):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_std, seed
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

    @classmethod
    def split(
        cls,
        dataset: "RegressionDataset",
        train_ratio: float = 0.8,
        seed: int | None = None,
    ) -> tuple[Dataset, Dataset]:
        """Return train/val splits."""
        total = len(dataset)
        train_len = int(total * train_ratio)
        return random_split(
            dataset, [train_len, total - train_len], generator=torch.Generator().manual_seed(seed)
        )


class QuantumRegression(nn.Module):
    """Deep regression network with optional dropout and batch‑norm."""

    def __init__(
        self,
        num_features: int,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)


__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
