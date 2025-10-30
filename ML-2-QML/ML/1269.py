"""Enhanced classical regression module with advanced architecture and utilities.

This module extends the original seed by adding:
- A configurable dense network with optional batch norm and dropout.
- A small utility class `RegressionTrainer` that wraps a PyTorch `LightningModule` for quick experimentation.
- Data normalization support within the dataset.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F
from typing import Optional, Tuple

def generate_superposition_data(num_features: int, samples: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data using a superposition-inspired target.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature space.
    samples : int
        Number of samples to generate.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    x : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Target vector of shape (samples,).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset returning a dictionary with ``states`` and ``target``.
    Supports optional standard‑score normalization.
    """
    def __init__(self, samples: int, num_features: int, normalize: bool = True, seed: Optional[int] = None):
        self.features, self.labels = generate_superposition_data(num_features, samples, seed)
        if normalize:
            self.mean = self.features.mean(axis=0, keepdims=True)
            self.std = self.features.std(axis=0, keepdims=True) + 1e-6
            self.features = (self.features - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Deep feed‑forward architecture with configurable depth, width, dropout, and batch‑norm.
    """
    def __init__(
        self,
        num_features: int,
        hidden_sizes: Tuple[int,...] = (64, 32),
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

class RegressionTrainer(nn.Module):
    """
    Lightweight wrapper that provides a ``train_step`` method and metrics for quick prototyping.
    This class is not a full LightningModule but mimics its interface for compatibility.
    """
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, batch):
        states, targets = batch["states"], batch["target"]
        preds = self.model(states)
        loss = self.criterion(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            preds = []
            targets = []
            for batch in dataloader:
                preds.append(self.model(batch["states"]))
                targets.append(batch["target"])
            preds = torch.cat(preds)
            targets = torch.cat(targets)
            mse = nn.functional.mse_loss(preds, targets).item()
            mae = nn.functional.l1_loss(preds, targets).item()
        self.model.train()
        return {"mse": mse, "mae": mae}

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
