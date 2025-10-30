"""
EstimatorQNN (classical) – a configurable feed‑forward regressor with training utilities.
"""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple, Optional


class EstimatorQNN(nn.Module):
    """
    A fully‑connected regression network with optional dropout and configurable depth.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : Iterable[int], optional
        Sequence of hidden layer sizes. Defaults to (32, 16).
    dropout : float, optional
        Dropout probability applied after each hidden layer. 0.0 disables dropout.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Iterable[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims) if hidden_dims is not None else [32, 16]
        layers: list[nn.Module] = []

        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    # ------------------------------------------------------------------ #
    # Training utilities
    # ------------------------------------------------------------------ #
    def train_on_dataset(
        self,
        dataset: TensorDataset,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        early_stop_patience: int | None = None,
        device: str = "cpu",
    ) -> Tuple[list[float], list[float]]:
        """
        Train the network on the supplied dataset.

        Returns
        -------
        train_losses, val_losses
        """
        self.to(device)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_losses: list[float] = []
        val_losses: list[float] = []

        best_val = float("inf")
        wait = 0

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(loader.dataset)
            train_losses.append(epoch_loss)

            # Simple validation by evaluating on the same data
            self.eval()
            with torch.no_grad():
                val_pred = self(dataset.tensors[0].to(device))
                val_loss = criterion(val_pred, dataset.tensors[1].to(device)).item()
                val_losses.append(val_loss)

            if early_stop_patience is not None and val_loss < best_val:
                best_val = val_loss
                wait = 0
            else:
                wait += 1

            if early_stop_patience is not None and wait >= early_stop_patience:
                break

        return train_losses, val_losses


__all__ = ["EstimatorQNN"]
