"""Classical feed‑forward classifier that mirrors the quantum helper interface with advanced training features."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class QuantumClassifier(nn.Module):
    """
    Classical feed‑forward classifier that mimics the quantum helper interface.
    The network is built with a configurable number of hidden layers (depth)
    and supports optional dropout, batch‑norm and early‑stopping during training.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 10,
        verbose: bool = False,
    ) -> None:
        """Train the network with early‑stopping based on validation loss."""
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        counter = 0
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)

            val_loss = self._eval_split(
                X, y, split=0.1, batch_size=batch_size, criterion=criterion, mode="val"
            )
            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs}  train={epoch_loss:.4f}  val={val_loss:.4f}"
                )

            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                torch.save(self.state_dict(), "_best_state.pt")
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break

        self.load_state_dict(torch.load("_best_state.pt"))

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return class labels (0/1) for the given inputs."""
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(X), dim=-1)[:, 1]
        return (probs >= threshold).long()

    def _eval_split(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        split: float,
        batch_size: int,
        criterion: nn.Module,
        mode: str = "val",
    ) -> float:
        """Internal helper to compute loss on a random split."""
        idx = torch.randperm(X.size(0))
        split_idx = int(split * X.size(0))
        if mode == "val":
            subset = idx[split_idx:]
        else:
            subset = idx[:split_idx]
        X_sub, y_sub = X[subset], y[subset]
        loader = DataLoader(TensorDataset(X_sub, y_sub), batch_size=batch_size)
        self.eval()
        loss_val = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.forward(xb)
                loss_val += criterion(logits, yb).item() * xb.size(0)
        return loss_val / len(loader.dataset)


__all__ = ["QuantumClassifier"]
