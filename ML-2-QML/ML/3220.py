"""Hybrid classical regression model that mirrors the quantum regression architecture while incorporating a learnable convolutional filter."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from Conv import Conv  # Classical convolutional filter from Conv.py


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with a sinusoidal relationship."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class HybridRegressionDataset(Dataset):
    """Dataset that provides both raw features and a 2×2 image representation."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        # Convert features to a 2×2 image by reshaping first 4 values
        self.images = self.features[:, :4].reshape(-1, 1, 2, 2)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[index], dtype=torch.float32),
            "image": torch.tensor(self.images[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegressionModel(nn.Module):
    """Purely classical baseline that first applies a convolutional filter to the image
    and then feeds the flattened vector into a deep feed‑forward network."""

    def __init__(self, num_features: int):
        super().__init__()
        self.conv_filter = Conv()  # 2×2 convolutional filter
        self.net = nn.Sequential(
            nn.Linear(num_features + 1, 32),  # +1 for the conv score
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute convolutional score
        conv_score = self.conv_filter.run(batch["image"].squeeze(0).numpy())
        conv_tensor = torch.tensor(conv_score, dtype=torch.float32, device=batch["features"].device)
        # Concatenate original features with conv score
        x = torch.cat([batch["features"], conv_tensor.unsqueeze(0)], dim=-1)
        return self.net(x).squeeze(-1)


__all__ = ["HybridRegressionDataset", "HybridRegressionModel", "generate_superposition_data"]
