"""Enhanced feed‑forward regressor with training utilities.

Features:
- Configurable depth, hidden sizes, dropout, and batch norm.
- Built‑in training loop with Adam optimizer and MSE loss.
- Evaluation method returning predictions as NumPy arrays.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class EstimatorNN(nn.Module):
    """A flexible fully‑connected regression network."""
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | tuple[int,...] = (16, 8),
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_model(
        self,
        X: torch.Tensor | list[float] | list[list[float]],
        y: torch.Tensor | list[float] | list[list[float]],
        *,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        """Train the network using Adam and MSE loss."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} loss={epoch_loss/len(loader):.4f}")

    def evaluate(self, X: torch.Tensor | list[float] | list[list[float]]) -> np.ndarray:
        """Return predictions as a NumPy array."""
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(next(self.parameters()).device)
            preds = self(X)
            return preds.cpu().numpy()

def EstimatorQNN() -> EstimatorNN:
    """Convenience factory mirroring the original API."""
    return EstimatorNN()

__all__ = ["EstimatorQNN"]
