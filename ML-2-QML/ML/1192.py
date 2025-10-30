"""QuantumClassifierModel: Classical classifier with advanced training features."""

from __future__ import annotations

from typing import Iterable, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class QuantumClassifierModel:
    """A classical feed‑forward classifier that mirrors the quantum interface.

    The network supports dropout, batch‑normalisation, optional residual
    connections and early‑stopping.  It exposes the same metadata
    (encoding, weight_sizes, observables) as the original seed so that
    downstream code can treat the classical and quantum models
    interchangeably.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        residual: bool = False,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        patience: int = 5,
        device: str = "cpu",
    ) -> None:
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(device)

        self.network, self.encoding, self.weight_sizes, self.observables = self._build_network()
        self.network.to(self.device)

    def _build_network(self) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        encoding = list(range(self.num_features))
        weight_sizes: List[int] = []

        for i in range(self.depth):
            linear = nn.Linear(in_dim, self.hidden_dim)
            layers.append(linear)
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            if self.residual and in_dim == self.hidden_dim:
                layers.append(nn.Identity())  # placeholder for residual
            in_dim = self.hidden_dim

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_ds = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.network.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.network(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)

            if val_loader is not None:
                val_loss = self._evaluate(val_loader, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.network.state_dict(), "best_model.pt")
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    self.network.load_state_dict(torch.load("best_model.pt"))
                    break

    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.network.eval()
        loss = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.network(xb)
                loss += criterion(logits, yb).item() * xb.size(0)
        return loss / len(loader.dataset)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            logits = self.network(
                torch.tensor(X, dtype=torch.float32).to(self.device)
            )
        return logits.argmax(dim=1).cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        )
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        loss = self._evaluate(test_loader, nn.CrossEntropyLoss())
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.network(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        accuracy = correct / total
        return loss, accuracy

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device))

    @property
    def metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return encoding, weight_sizes, observables for compatibility."""
        return self.encoding, self.weight_sizes, self.observables


__all__ = ["QuantumClassifierModel"]
