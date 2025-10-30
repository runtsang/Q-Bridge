"""Advanced classical regression model and dataset.

This module extends the original seed by adding feature
normalisation, configurable hidden layers, dropout, and a
flexible activation schedule.  The model can be used
directly in standard PyTorch training loops.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int,
                                noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the target.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(scale=noise_std, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset yielding state vectors and targets for regression."""
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """Feed‑forward regression network with optional dropout and batch‑norm."""
    def __init__(
        self,
        num_features: int,
        hidden_sizes: list[int] | tuple[int,...] = (32, 16),
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_features = h
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

    def freeze(self) -> None:
        """Freeze all parameters, useful when fine‑tuning."""
        for p in self.parameters():
            p.requires_grad = False


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
