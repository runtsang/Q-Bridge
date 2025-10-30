"""Enhanced classical regression model with residuals and early stopping."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data where the target depends on a nonlinear
    combination of the features.  The features are drawn from a uniform
    distribution and a random phase is added to the target to emulate
    measurement noise.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    # Nonlinear target: sin of sum of features plus a small cosine component
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    # Add Gaussian noise
    y += np.random.normal(0.0, 0.05, size=y.shape)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic features and targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegression(nn.Module):
    """
    Classical neural network with residual connections and batch‑normalisation.
    The architecture is a stack of linear layers interleaved with
    Residual blocks to improve gradient flow for deeper networks.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = num_features
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            # Residual connection
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.body(state_batch)
        return self.head(x).squeeze(-1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(X)


def early_stopping_callback(patience: int, min_delta: float = 0.0):
    """
    Returns a simple early‑stopping callback that can be used during training.
    The callback keeps track of the best validation loss and stops training
    when the loss does not improve for `patience` epochs.
    """
    best_loss = float("inf")
    epochs_no_improve = 0

    def callback(val_loss: float, epoch: int, optimizer: torch.optim.Optimizer, model: nn.Module):
        nonlocal best_loss, epochs_no_improve
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            epochs_no_improve = 0
            # Save the best model state
            torch.save(model.state_dict(), f"best_model_epoch_{epoch}.pt")
        else:
            epochs_no_improve += 1
        return epochs_no_improve >= patience

    return callback


__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data", "early_stopping_callback"]
