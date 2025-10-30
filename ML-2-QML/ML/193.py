"""Hybrid classical regression with feature selection and checkpointing."""

from __future__ import annotations

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

__all__ = ["HybridMLModel", "RegressionDataset", "generate_superposition_data"]

def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic dataset based on superposition angles.
    Returns tensors of shape (samples, num_features) and (samples,).
    """
    rng = np.random.default_rng()
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return torch.from_numpy(x), torch.from_numpy(y)

class RegressionDataset(Dataset):
    """Dataset wrapper yielding feature tensors and targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {"states": self.features[idx], "target": self.labels[idx]}

    def split(self, test_size: float = 0.2, random_state: int | None = None):
        """Return train/validation indices."""
        return train_test_split(
            torch.arange(len(self), dtype=torch.long),
            test_size=test_size,
            random_state=random_state,
        )

class HybridMLModel(nn.Module):
    """MLP that optionally performs early‑stopping and checkpointing."""
    def __init__(self, num_features: int, hidden_sizes: list[int] | None = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 16]
        layers = [nn.Linear(num_features, hidden_sizes[0]), nn.ReLU()]
        for h in hidden_sizes[1:]:
            layers.append(nn.Linear(hidden_sizes[0], h))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.backbone(x)).squeeze(-1)

    def train_one_epoch(self, loader, optimizer, criterion):
        self.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            preds = self.forward(batch["states"])
            loss = criterion(preds, batch["target"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def save_checkpoint(self, path: str):
        """Persist model weights."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path: str, **kwargs):
        """Load weights into a new model instance."""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        return model
