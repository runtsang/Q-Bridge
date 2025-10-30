"""Enhanced classical regression model with an attention mask and validation split."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data resembling a quantum superposition pattern.
    Each sample is a vector of length ``num_features``.
    The label is a non‑linear sinusoidal function of the sum of the features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.10 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns input vectors and target scalars.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

def split_dataset(dataset: Dataset, val_ratio: float = 0.2) -> tuple[Dataset, Dataset]:
    """
    Split a dataset into training and validation subsets.
    """
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    return random_split(dataset, [n_train, n_val])

class QModelHybrid(nn.Module):
    """
    Classical regression model with an attention mask learning feature importance.
    The mask is a trainable parameter passed through a sigmoid to keep values in (0,1).
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(num_features))
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        masked = state_batch * torch.sigmoid(self.mask)
        return self.net(masked).squeeze(-1)

    @staticmethod
    def hybrid_loss(classical_pred: torch.Tensor,
                    quantum_pred: torch.Tensor,
                    target: torch.Tensor,
                    alpha: float = 0.5) -> torch.Tensor:
        """
        Weighted mean‑square error between predictions and target.
        """
        mse = nn.MSELoss()
        return alpha * mse(classical_pred, target) + (1.0 - alpha) * mse(quantum_pred, target)

__all__ = ["QModelHybrid", "RegressionDataset", "generate_superposition_data", "split_dataset"]
