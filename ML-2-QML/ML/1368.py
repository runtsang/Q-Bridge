"""Enhanced classical regression pipeline with train/validation split,
early stopping, and optimizer flexibility."""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.05,
    shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where the target is a non‑linear
    function of the feature sum, optionally perturbed by Gaussian noise.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise. Default is 0.05.
    shift : float, optional
        Constant shift applied to the target. Default is 0.0.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Target vector of shape (samples,).
    """
    rng = np.random.default_rng()
    X = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles) + shift
    y += rng.normal(scale=noise_std, size=samples).astype(np.float32)
    return X, y


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(torch.utils.data.Dataset):
    """
    Torch dataset wrapping the synthetic superposition data.
    """

    def __init__(self, samples: int, num_features: int, noise_std: float = 0.05, shift: float = 0.0):
        self.features, self.labels = generate_superposition_data(
            num_features=num_features,
            samples=samples,
            noise_std=noise_std,
            shift=shift,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class RegressionModel(nn.Module):
    """
    Configurable MLP for regression with optional batch‑norm and dropout.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: Iterable[int] = (64, 32),
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        batch_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
@dataclass
class EarlyStopping:
    """
    Simple early‑stopping wrapper based on validation loss.
    """
    patience: int = 10
    min_delta: float = 1e-4
    _best_loss: float | None = None
    _patience_counter: int = 0

    def step(self, val_loss: float) -> bool:
        if self._best_loss is None or val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._patience_counter = 0
            return False
        self._patience_counter += 1
        return self._patience_counter >= self.patience


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        states = batch["states"].to(device)
        target = batch["target"].to(device)
        optimizer.zero_grad()
        pred = model(states)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * states.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            target = batch["target"].to(device)
            preds.append(model(states).cpu())
            trues.append(target.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    mse = mean_squared_error(trues, preds)
    r2 = r2_score(trues, preds)
    return mse, r2


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data", "EarlyStopping"]
