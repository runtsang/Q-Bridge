"""Extended classical regression model with batch‑normalisation, dropout, and training utilities.

The module mirrors the original seed but adds richer neural architecture and
an optional `fit` method that trains the network on a PyTorch DataLoader.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, Tuple, Optional


def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input space.
    samples : int
        Number of samples to generate.

    Returns
    -------
    x : np.ndarray
        Input features of shape (samples, num_features).
    y : np.ndarray
        Scalar targets of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the synthetic data.
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
    Classical regression network with optional dropout and batch‑norm.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    dropout_rate : float, optional
        Dropout probability; set to 0.0 to disable.
    """

    def __init__(self, num_features: int, dropout_rate: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

    # ------------------------------------------------------------------
    # Convenience training helper
    # ------------------------------------------------------------------
    def fit(
        self,
        dataloader: Iterable[dict],
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> list[float]:
        """
        Train the network using MSE loss.

        Parameters
        ----------
        dataloader : Iterable[dict]
            Iterable yielding batches of ``{'states': Tensor, 'target': Tensor}``.
        epochs : int
            Number of epochs.
        lr : float
            Learning rate.
        weight_decay : float
            L2 regularisation strength.
        device : torch.device, optional
            Device to run on; defaults to CUDA if available.

        Returns
        -------
        losses : list[float]
            Training loss after each epoch.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        losses: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                states, target = batch["states"].to(device), batch["target"].to(device)
                optimizer.zero_grad()
                pred = self(states)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * states.size(0)
            losses.append(epoch_loss / len(dataloader.dataset))
        return losses

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return predictions for a batch of inputs.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(X)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
