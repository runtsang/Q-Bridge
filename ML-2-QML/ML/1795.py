\
"""Enhanced classical regression model with training utilities."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Sequence, Optional

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of training examples.

    Returns
    -------
    features : np.ndarray
        Shape (samples, num_features), float32.
    labels : np.ndarray
        Shape (samples,), float32.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    Dataset returning a dict with ``states`` (features) and ``target`` (label).

    Supports an optional transformation applied to the feature tensor.
    """
    def __init__(self, samples: int, num_features: int, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return {"states": x, "target": y}

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class QuantumRegressionModel(nn.Module):
    """
    Fully‑connected regression network with batch‑norm, dropout and L2 regularisation.
    """
    def __init__(
        self,
        num_features: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.2,
        l2: float = 1e-4,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.l2 = l2

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def train_model(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Train ``model`` on ``dataset`` using MSE loss and Adam optimiser.

    Returns the trained model.
    """
    device = torch.device(device or "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            states = batch["states"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad()
            preds = model(states)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        scheduler.step()
        if verbose:
            print(f"[{epoch}/{epochs}] loss: {epoch_loss/len(dataset):.4f}")
    return model

def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> float:
    """
    Return the mean‑squared error on ``dataset``.
    """
    device = torch.device(device or "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        total, count = 0.0, 0
        for batch in loader:
            states = batch["states"].to(device)
            targets = batch["target"].to(device)
            preds = model(states)
            total += criterion(preds, targets).item() * states.size(0)
            count += states.size(0)
    return total / count

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "QuantumRegressionModel",
    "train_model",
    "evaluate_model",
]
