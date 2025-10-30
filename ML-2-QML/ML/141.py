"""Enhanced classical regression model with flexible architecture and training utilities.

This module mirrors the structure of the original seed but adds
- a residual‑style MLP that can be configured with arbitrary hidden layers,
- dropout for regularization,
- a simple training loop with early stopping,
- GPU support and evaluation helpers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where the target is a non‑linear
    function of the input features.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of data points.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the target.

    Returns
    -------
    X, y : np.ndarray
        Feature matrix and target vector.
    """
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += np.random.normal(scale=noise_std, size=y.shape)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch dataset that holds the synthetic regression data.
    """
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.05):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ResidualBlock(nn.Module):
    """
    A small residual block used in the MLP.
    """
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.bn(self.fc(x)))

class QModel(nn.Module):
    """
    Flexible MLP with optional residual connections and dropout.

    Parameters
    ----------
    num_features : int
        Size of input vector.
    hidden_dims : Sequence[int], optional
        List of hidden layer sizes. Defaults to [64, 32].
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    use_residual : bool, optional
        If True, each hidden layer is wrapped in a ResidualBlock.
    """
    def __init__(
        self,
        num_features: int,
        hidden_dims: list[int] | tuple[int,...] = (64, 32),
        dropout: float = 0.0,
        use_residual: bool = False,
    ):
        super().__init__()
        layers = []
        in_dim = num_features
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            if use_residual:
                layers.append(ResidualBlock(dim, dropout))
            in_dim = dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

    @staticmethod
    def train_loop(
        model: "QModel",
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int,
        lr: float,
        device: torch.device,
        early_stop_patience: int = 10,
    ) -> tuple["QModel", list[float], list[float]]:
        """
        Simple training loop with early stopping based on validation MSE.

        Returns
        -------
        model, train_losses, val_losses
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_losses, val_losses = [], []
        best_val = float("inf")
        patience = 0
        model.to(device)
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                preds = model(batch["states"].to(device))
                loss = criterion(preds, batch["target"].to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch["states"].size(0)
            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)

            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        preds = model(batch["states"].to(device))
                        loss = criterion(preds, batch["target"].to(device))
                        val_loss += loss.item() * batch["states"].size(0)
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                    torch.save(model.state_dict(), "best_model.pt")
                else:
                    patience += 1
                    if patience >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                val_losses.append(float("nan"))

        # Load best weights
        if val_loader is not None:
            model.load_state_dict(torch.load("best_model.pt"))
        return model, train_losses, val_losses

    def evaluate(self, loader: DataLoader, device: torch.device) -> float:
        """
        Compute mean squared error on the given loader.
        """
        self.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                preds = self(batch["states"].to(device))
                loss = criterion(preds, batch["target"].to(device))
                total_loss += loss.item() * batch["states"].size(0)
        return total_loss / len(loader.dataset)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
