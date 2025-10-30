"""Enhanced classical estimator mirroring EstimatorQNN with advanced training utilities.

Features:
  • Configurable hidden layers, dropout, batch‑norm.
  • sklearn‑style `fit`, `predict`, and `score` methods.
  • Early‑stopping and GPU support.
  • Optional data‑scaling via StandardScaler.

The class remains importable as EstimatorQNN and can replace the seed model
in downstream experiments while providing richer diagnostics.
"""

import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


class EstimatorQNN(nn.Module):
    """Classical feed‑forward regression network with training utilities."""
    def __init__(
        self,
        input_dim: int = 2,
        hidden_sizes: tuple[int,...] = (8, 4),
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 1000,
        batch_size: int = 32,
        early_stop_patience: int = 10,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.scaler = StandardScaler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------------
    #  Training utilities
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EstimatorQNN":
        """Fit the network on the supplied data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        # Scale inputs
        X = self.scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.early_stop_patience:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for the supplied data."""
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.eval()
        with torch.no_grad():
            preds = self(X).cpu().numpy().flatten()
        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score for the predictions."""
        preds = self.predict(X)
        return r2_score(y, preds)


__all__ = ["EstimatorQNN"]
