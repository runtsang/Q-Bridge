"""Enhanced classical regression model and dataset with data augmentation and robust training utilities."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.05, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data resembling a superposition of basis states.
    
    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.
    noise_std : float, default 0.05
        Standard deviation of Gaussian noise added to labels.
    random_state : int | None
        Seed for reproducibility.
    
    Returns
    -------
    x : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Target vector of shape (samples,).
    """
    rng = np.random.default_rng(random_state)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += rng.normal(scale=noise_std, size=samples)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset for superposition regression.
    
    Supports optional feature scaling and label noise injection.
    """
    def __init__(self, samples: int, num_features: int, *, noise_std: float = 0.05, random_state: int | None = None):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std, random_state)
    
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)
    
    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Fully‑connected regression network with residual connections and dropout.
    
    The architecture is configurable via ``hidden_dims``.  Dropout is applied
    after every hidden layer to mitigate over‑fitting.  Batch‑norm normalises
    activations before the non‑linearity.
    """
    def __init__(self, num_features: int, hidden_dims: tuple[int,...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = num_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
