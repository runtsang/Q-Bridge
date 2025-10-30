"""Enhanced classical regression model with residual blocks and data‑augmentation support."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic data where the target depends on the sum of input features.
    The data is perturbed with Gaussian noise and optionally scaled.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    # Inject Gaussian noise to emulate measurement uncertainty
    y += 0.05 * np.random.randn(samples).astype(np.float32)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset exposing input features and target values.
    Optional scaling can be applied at construction.
    """
    def __init__(self, samples: int, num_features: int, scale: bool = True):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.scaler = StandardScaler() if scale else None
        if self.scaler:
            self.features = self.scaler.fit_transform(self.features).astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """
    A simple residual block with two linear layers, batch‑norm and ReLU.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.lin1(x)))
        out = self.dropout(out)
        out = self.bn2(self.lin2(out))
        out += residual
        return F.relu(out)


class QRegressionModel(nn.Module):
    """
    Classical regression network with residual blocks, dropout, and a final linear head.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.res1 = ResidualBlock(num_features, 64, dropout=0.2)
        self.res2 = ResidualBlock(64, 32, dropout=0.2)
        self.head = nn.Linear(32, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.res1(state_batch)
        out = self.res2(out)
        return self.head(out).squeeze(-1)


__all__ = ["QRegressionModel", "RegressionDataset", "generate_superposition_data"]
