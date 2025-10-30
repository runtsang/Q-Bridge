"""Enhanced classical regression model with feature scaling and dropout."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    """Dataset that generates noisy superposition‑style regression data."""

    def __init__(self, samples: int, num_features: int, noise_std: float = 0.05):
        self.features, self.labels = self._generate_data(num_features, samples, noise_std)

    def _generate_data(self, num_features, samples, noise_std):
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        y += np.random.normal(0, noise_std, size=y.shape)
        return x, y.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QModel(nn.Module):
    """Deep feed‑forward network with batch‑norm, ReLU and dropout."""

    def __init__(self, num_features: int, hidden_sizes: tuple[int,...] = (64, 32)):
        super().__init__()
        layers = []
        input_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


__all__ = ["QModel", "RegressionDataset"]
