"""Enhanced classical regression model with early‑stopping and ensembling.

This module extends the original seed by adding reproducible data generation,
an early‑stopping mechanism, and a simple ensemble wrapper for the regression
network.  The network architecture remains a lightweight feed‑forward net
but exposes a flexible interface for experimentation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Iterable, List, Tuple

def generate_superposition_data(
    num_features: int,
    samples: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data that mimics the superposition‑like pattern of the original seed.

    Parameters
    ----------
    num_features : int
        Number of independent features per sample.
    samples : int
        Number of samples to generate.
    seed : int | None, optional
        Random seed for reproducibility.  If ``None`` the generator is left
        un‑seeded.

    Returns
    -------
    X : np.ndarray, shape (samples, num_features)
        Random features drawn from a uniform distribution on [-1, 1].
    y : np.ndarray, shape (samples,)
        Target values computed as ``sin(sum(x)) + 0.1*cos(2*sum(x))``.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    ``torch.utils.data.Dataset`` wrapper around the synthetic superposition data.
    """

    def __init__(self, samples: int, num_features: int, seed: int | None = None):
        self.features, self.labels = generate_superposition_data(num_features, samples, seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression__gen226(nn.Module):
    """
    Classical feed‑forward regression network.

    The architecture is deliberately simple: two hidden layers with ReLU
    activations.  The class is exposed as ``QuantumRegression__gen226`` to
    mirror the quantum counterpart, so that both modules can be loaded
    interchangeably in downstream experiments.
    """

    def __init__(self, num_features: int, hidden: Tuple[int,...] = (32, 16)):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

class EarlyStopping:
    """
    Simple early‑stopping utility that monitors a validation metric.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs with no improvement before stopping.
    min_delta : float
        Minimum change in the monitored value to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, current_score: float) -> None:
        if self.best_score is None or current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True

class EnsembleModel(nn.Module):
    """
    Simple ensemble wrapper that averages predictions from a list of models.

    Each model in ``models`` is expected to produce a 1‑D output of shape
    ``(batch_size,)``.  The ensemble returns the arithmetic mean of these
    outputs.
    """

    def __init__(self, models: Iterable[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        preds = [m(state_batch) for m in self.models]
        return torch.stack(preds, dim=0).mean(dim=0)

__all__ = [
    "QuantumRegression__gen226",
    "RegressionDataset",
    "generate_superposition_data",
    "EarlyStopping",
    "EnsembleModel",
]
