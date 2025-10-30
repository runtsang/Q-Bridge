"""Enhanced classical regression model with residual connections and data augmentation.

This module mirrors the original `QuantumRegression.py` but expands the
dataset generator and model architecture.  The `RegressionDataset` now
supports optional Gaussian noise and can be instantiated with a
different number of features.  The neural network `QModel` employs a
residual block, batch‑norm, dropout, and a configurable hidden layer
size, providing a richer function approximator while keeping the
public API identical to the seed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.
    The target is a smooth non‑linear function of the feature sum
    with optional Gaussian noise.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    X : np.ndarray of shape (samples, num_features)
    y : np.ndarray of shape (samples,)
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    # Base signal: sin(2πθ) + 0.1*cos(4πθ)
    y = np.sin(2 * np.pi * angles) + 0.1 * np.cos(4 * np.pi * angles)
    if noise_std > 0.0:
        y += np.random.normal(0.0, noise_std, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset for the synthetic regression problem."""
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ResidualBlock(nn.Module):
    """A simple residual block used inside `QModel`."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.ReLU(inplace=True)(self.layer(x) + self.shortcut(x))

class QModel(nn.Module):
    """
    Classical regression model with a residual architecture.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    hidden_sizes : list[int], optional
        Sizes of the hidden layers.  Defaults to [64, 32].
    dropout : float, optional
        Dropout probability applied after each residual block.
    """
    def __init__(self, num_features: int, hidden_sizes: list[int] | None = None, dropout: float = 0.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        layers = []
        in_features = num_features
        for size in hidden_sizes:
            layers.append(ResidualBlock(in_features, size, dropout))
            in_features = size
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

    def predict(self, state_batch: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Convenience wrapper for inference."""
        self.eval()
        with torch.no_grad():
            if device is not None:
                state_batch = state_batch.to(device)
            return self.forward(state_batch)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
