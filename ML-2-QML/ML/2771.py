"""Classical hybrid regression model inspired by the original seeds.

The model is a pure‑classical feed‑forward network that can be paired
with the quantum version to form a hybrid pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data that matches the quantum dataset.

    The data distribution is based on the same sinusoidal target used
    in the quantum seed: y = sin(∑x) + 0.1 cos(2∑x).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns feature vectors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridRegression(nn.Module):
    """Pure‑classical feed‑forward regressor.

    The architecture mirrors the original `QModel` but uses only
    PyTorch primitives.  It can be paired with the quantum model
    in a separate training loop to form a hybrid pipeline.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
