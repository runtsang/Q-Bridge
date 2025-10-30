from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset where each sample is a
    superposition of two orthogonal basis states.  The labels are a
    smooth, nonlinear function of the sum of the input angles.
    """
    # Uniform angles per feature
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns a dictionary with'states' and 'target' tensors."""
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
    """A lightweight residual block that can be stacked to increase depth."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return self.bn(x + out)


class QModel(nn.Module):
    """A deeper, residual‑augmented MLP that still uses the same feature dimension."""
    def __init__(self, num_features: int, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(num_features, 32)]
        # Stack residual blocks
        for _ in range(depth):
            layers.append(ResidualBlock(32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(32, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
