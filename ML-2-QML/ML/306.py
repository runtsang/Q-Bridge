"""Enhanced classical regressor with configurable depth, regularisation, and training helpers."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple, List, Callable, Optional


class EnhancedEstimatorQNN(nn.Module):
    """
    Fully‑connected regression network with optional batch‑norm and dropout.

    Parameters
    ----------
    input_dim : int
        Size of the input feature vector.
    hidden_dims : List[int]
        Sizes of hidden layers.
    dropout_rate : float, default=0.0
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: List[int] = [8, 4],
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # regression output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

    # ------------------------------------------------------------------
    # Training utilities – not required for inference but useful for experiments
    # ------------------------------------------------------------------
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stop_patience: int = 10,
        device: str = "cpu",
        verbose: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """
        Train the network using MSE loss and Adam optimizer.

        Returns training and validation loss history.
        """
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        patience_counter = 0
        train_losses: List[float] = []
        val_losses: List[float] = []

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = self(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        preds = self(xb)
                        loss = criterion(preds, yb)
                        val_loss += loss.item() * xb.size(0)
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)

                if verbose:
                    print(f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        if verbose:
                            print("Early stopping triggered.")
                        break
            else:
                if verbose:
                    print(f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.5f}")

        return train_losses, val_losses

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return predictions for input tensor X."""
        self.eval()
        with torch.no_grad():
            return self(X)


def EstimatorQNN(
    input_dim: int = 2,
    hidden_dims: List[int] = [8, 4],
    dropout_rate: float = 0.0,
) -> EnhancedEstimatorQNN:
    """
    Factory function to instantiate the enhanced regressor.

    The function signature mirrors the original seed for backward compatibility.
    """
    return EnhancedEstimatorQNN(input_dim, hidden_dims, dropout_rate)


__all__ = ["EnhancedEstimatorQNN", "EstimatorQNN"]
