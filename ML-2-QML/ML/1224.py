"""Enhanced classical regression model and dataset."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce synthetic regression data by sampling random feature vectors
    and applying a nonlinear transformation.  The function now adds Gaussian
    noise to the labels to emulate measurement uncertainty.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    noise = np.random.normal(scale=0.05, size=y.shape).astype(np.float32)
    return x, (y + noise).astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the synthetic regression data.
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


class QModel(nn.Module):
    """
    Feed‑forward regression network with residual connections, batch‑norm,
    and dropout.  The architecture is intentionally modular to allow
    easy experimentation with hidden layer sizes.
    """
    def __init__(self, num_features: int, hidden_dims: list[int] = [64, 32, 16], dropout: float = 0.2):
        super().__init__()
        layers = []
        input_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.net(state_batch)
        return self.output(x).squeeze(-1)

    def predict(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(state_batch)

    def mse(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute mean‑squared error between predictions and targets.
        """
        return torch.mean((predictions - targets) ** 2)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
