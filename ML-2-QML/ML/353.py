"""
Enhanced classical feed‑forward regressor for regression tasks.
Provides training, prediction, evaluation, and weight persistence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
import numpy as np


class EstimatorQNN(nn.Module):
    """
    Deep residual neural network with dropout and batch‑norm.
    Architecture: 2 → 16 → 32 → 16 → 1 (with residual skip from input to hidden layers).
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm_input = nn.BatchNorm1d(2)
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
        self.residual = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm_input(x)
        out = self.net(x_norm)
        out += self.residual(x_norm)
        return out

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Train the network using MSE loss and Adam optimizer.
        """
        device = device or torch.device("cpu")
        self.to(device)

        # Normalise data
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-8
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8

        X_norm = (X - self.X_mean) / self.X_std
        y_norm = (y - self.y_mean) / self.y_std

        dataset = TensorDataset(torch.from_numpy(X_norm).float(),
                                torch.from_numpy(y_norm).float().unsqueeze(1))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.forward(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} – MSE: {epoch_loss:.4f}")

    # ------------------------------------------------------------------
    # Prediction & evaluation
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
        """
        Return predictions in original scale.
        """
        device = device or torch.device("cpu")
        self.eval()
        with torch.no_grad():
            X_norm = (X - self.X_mean) / self.X_std
            preds = self.forward(torch.from_numpy(X_norm).float().to(device))
            preds = preds.cpu().numpy().squeeze()
            return preds * self.y_std + self.y_mean

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Return MAE and RMSE on the provided data.
        """
        preds = self.predict(X)
        mae = np.mean(np.abs(preds - y))
        rmse = np.sqrt(np.mean((preds - y) ** 2))
        return mae, rmse

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

__all__ = ["EstimatorQNN"]
