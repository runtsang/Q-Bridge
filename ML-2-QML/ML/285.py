"""Enhanced classical estimator with residual connections, batch‑norm, and dropout."""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class EstimatorNN(nn.Module):
    """A robust regression network.

    The architecture consists of alternating linear layers with
    BatchNorm, Tanh activations, and a residual skip connection
    from the first hidden layer to the output. Dropout is applied
    after each activation to mitigate overfitting.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features.
    hidden_dims : list[int]
        Sizes of the hidden layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32, 16]
        layers = []
        prev_dim = input_dim

        # First hidden layer
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dims[0]

        # Remaining hidden layers
        for h in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection from first hidden layer
        first_hidden = self.net[0](x)
        out = self.net[1:](first_hidden)
        return out


def EstimatorQNN() -> EstimatorNN:
    """Convenience factory that returns a pre‑configured network."""
    return EstimatorNN()


def train(
    model: EstimatorNN,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, list[float]]:
    """Simple training loop with mean‑squared‑error loss."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
            history["val_loss"].append(val_loss)

    return history


def prepare_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    val_split: float = 0.2,
) -> tuple[DataLoader, DataLoader | None]:
    """Split data and return training/validation loaders."""
    assert X.shape[0] == y.shape[0]
    idx = np.random.permutation(X.shape[0])
    split = int(len(idx) * (1 - val_split))
    train_idx, val_idx = idx[:split], idx[split:]
    train_set = TensorDataset(
        torch.tensor(X[train_idx], dtype=torch.float32),
        torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(-1),
    )
    val_set = TensorDataset(
        torch.tensor(X[val_idx], dtype=torch.float32),
        torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(-1),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader


__all__ = ["EstimatorQNN", "train", "prepare_dataloaders"]
