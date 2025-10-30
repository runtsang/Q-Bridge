"""Enhanced classical regression model with training utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import functional as F
from typing import Iterable, Tuple


def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where the target is a
    non‑linear function of the sum of input features.

    Parameters
    ----------
    num_features : int
        Dimensionality of each sample.
    samples : int
        Number of samples to generate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Features ``(samples, num_features)`` and targets ``(samples,)``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the synthetic regression data.
    """

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
    """
    Classical neural network with residual connection, batch‑norm and dropout.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # Residual shortcut
        self.residual = nn.Linear(num_features, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass with residual addition.
        """
        out = self.net(state_batch)
        res = self.residual(state_batch)
        return out + res

    # Training utilities ----------------------------------------------------
    def fit(
        self,
        train_loader: Iterable[dict],
        val_loader: Iterable[dict] | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Simple training loop with optional validation.

        Parameters
        ----------
        train_loader : Iterable[dict]
            DataLoader yielding batches of ``{states, target}``.
        val_loader : Iterable[dict] | None
            Optional validation DataLoader.
        epochs : int
            Number of epochs.
        lr : float
            Learning rate.
        device : torch.device | str
            Device to run on.
        """
        self.to(device)
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            for batch in train_loader:
                states = batch["states"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad()
                pred = self.forward(states)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            if val_loader:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        states = batch["states"].to(device)
                        target = batch["target"].to(device)
                        pred = self.forward(states)
                        val_loss += criterion(pred, target).item()
                val_loss /= len(val_loader)
                print(f"Epoch {epoch:02d} | Train loss: {avg_loss:.4f} | Val loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch:02d} | Train loss: {avg_loss:.4f}")

    def predict(self, loader: Iterable[dict], device: torch.device | str = "cpu") -> torch.Tensor:
        """
        Predict on a dataset.

        Parameters
        ----------
        loader : Iterable[dict]
            DataLoader yielding batches.
        device : torch.device | str
            Device to run on.

        Returns
        -------
        torch.Tensor
            Concatenated predictions.
        """
        self.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                states = batch["states"].to(device)
                preds.append(self.forward(states))
        return torch.cat(preds, dim=0)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
