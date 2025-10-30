"""
Enhanced classical feed‑forward regressor.

Features
--------
* Min‑max feature scaling.
* Two hidden layers with ReLU activations.
* Dropout regularisation.
* Residual connection between input and output.
* `fit`/`predict` interface compatible with scikit‑learn.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


class EstimatorQNN(BaseEstimator, RegressorMixin):
    """
    A lightweight neural network regressor that mirrors the original EstimatorQNN
    but adds scaling, dropout, and a residual connection.
    """

    def __init__(
        self,
        hidden_sizes: Tuple[int, int] = (16, 8),
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.scaler = MinMaxScaler()
        self._build_model()

    def _build_model(self) -> None:
        layers = [
            nn.Linear(2, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_sizes[1], 1),
        ]
        self.model = nn.Sequential(*layers).to(self.device)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "EstimatorQNN":
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        X_scaled = self.scaler.fit_transform(X_np)
        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                residual = xb
                out = self.model(xb)
                out = out + residual[:, :1]  # residual from first feature
                loss = criterion(out, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X_np = X.cpu().numpy()
        X_scaled = self.scaler.transform(X_np)
        xb = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(xb)
            out = out + xb[:, :1]  # residual
        return out.cpu()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hidden_sizes={self.hidden_sizes}, dropout={self.dropout})"


__all__ = ["EstimatorQNN"]
