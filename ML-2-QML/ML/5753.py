"""Classical regression model with enhanced regularisation and residual connections."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.

    Args:
        num_features: Dimensionality of the feature vector.
        samples: Number of samples to generate.

    Returns:
        Tuple of feature matrix (float32) and target vector (float32).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles) + 0.05 * np.random.randn(samples)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding feature tensors and continuous targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """MLP with dropout, batch‑norm and a residual skip connection."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.residual = nn.Linear(num_features, 32)
        self.head = nn.Linear(32, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.net(state_batch)
        res = self.residual(state_batch)
        x = x + res
        return self.head(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
