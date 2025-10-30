"""QuantumRegression__gen282 module – classical regression with residual layers.

The module keeps the original public symbols (`QModel`, `RegressionDataset`,
`generate_superposition_data`) but adds:
* a noise parameter to the data generator,
* a residual MLP that can be configured for depth and hidden size,
* a lightweight `train_one_epoch` helper that can be dropped into any training loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Callable

def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    noise: float = 0.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a superposition‑based regression dataset.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.
    noise : float, optional
        Standard deviation of additive Gaussian noise added to targets.
    random_state : int | None, optional
        Seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise > 0.0:
        y += rng.normal(scale=noise, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Torch dataset that yields feature vectors and scalar targets."""
    def __init__(self, samples: int, num_features: int, noise: float = 0.0, random_state: int | None = None):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise=noise, random_state=random_state
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Residual MLP that mirrors the quantum model’s input size."""
    def __init__(self, num_features: int, hidden: int = 64, depth: int = 4):
        super().__init__()
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden, in_dim))
            layers.append(nn.ReLU())
        self.res_blocks = nn.Sequential(*layers)
        self.head = nn.Linear(num_features, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.res_blocks(state_batch)
        return self.head(x).squeeze(-1)

def train_one_epoch(
    model: QModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
    device: torch.device | str | None = None,
) -> float:
    """Run a single training epoch and return the average loss."""
    model.train()
    device = device or torch.device("cpu")
    total_loss = 0.0
    for batch in loader:
        states = batch["states"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad()
        preds = model(states)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * states.size(0)
    return total_loss / len(loader.dataset)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data", "train_one_epoch"]
