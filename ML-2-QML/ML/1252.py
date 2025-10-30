"""Extended classical regression model with data handling utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

def generate_superposition_data(num_features: int, samples: int, noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data where the target is a nonlinear function of the sum of features.
    Optionally add Gaussian noise.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise > 0.0:
        y += np.random.normal(scale=noise, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch Dataset wrapping the synthetic regression data.
    Supports optional feature transforms.
    """
    def __init__(self, samples: int, num_features: int, transform: Any | None = None):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.transform = transform
        if self.transform is not None:
            self.features = self.transform(self.features)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

    def get_dataloaders(self, batch_size: int, split: float = 0.8, shuffle: bool = True) -> Dict[str, DataLoader]:
        """
        Split the dataset into train/validation loaders.
        """
        n_train = int(len(self) * split)
        train_ds, val_ds = torch.utils.data.random_split(self, [n_train, len(self) - n_train])
        return {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        }

class QModel(nn.Module):
    """
    Feed‑forward neural network with optional batch normalization and dropout.
    """
    def __init__(self, num_features: int, hidden_dims: Tuple[int,...] = (32, 16), dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

    def training_step(self, batch: Dict[str, torch.Tensor], criterion: nn.Module) -> torch.Tensor:
        preds = self(batch["states"])
        loss = criterion(preds, batch["target"])
        return loss

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            preds = self(batch["states"])
            return preds, batch["target"]

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
