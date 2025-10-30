"""
Enhanced classical feed‑forward regressor with regularization and a built‑in training pipeline.
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

class EstimatorQNN(nn.Module):
    """
    Fully‑connected neural network with batch‑norm, dropout and an `fit` / `predict`
    interface for quick experimentation.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | tuple[int,...] = (16, 32),
                 dropout: float = 0.2) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 200, batch_size: int = 32,
            lr: float = 1e-3, patience: int = 10) -> None:
        """
        Train the network using MSE loss with Adam optimiser and early stopping.
        """
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_loss = float("inf")
        counter = 0
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.state_dict(), "best_model.pt")
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.load_state_dict(torch.load("best_model.pt"))

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

def EstimatorQNN() -> EstimatorQNN:
    """Return an untrained instance of the enhanced neural network."""
    return EstimatorQNN()

__all__ = ["EstimatorQNN"]
