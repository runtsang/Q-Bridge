"""QuantumRegression__gen236_ml.py

Enhanced classical regression module with richer dataset generation and a deeper neural network.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    Features are sampled from a uniform distribution over [-π, π] and the labels are a
    nonlinear combination of sin and cos of the features, with optional Gaussian noise.
    """
    x = np.random.uniform(-np.pi, np.pi, size=(samples, num_features)).astype(np.float32)
    y = np.sin(x).sum(axis=1) + 0.5 * np.cos(x).sum(axis=1)
    y += noise_std * np.random.randn(samples).astype(np.float32)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset class for the classical regression task."""

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
    """Deep neural network for regression with residual‑style connections, batch‑norm and dropout."""

    def __init__(self, num_features: int, hidden_dims: tuple[int,...] = (64, 32)):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.1))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)
