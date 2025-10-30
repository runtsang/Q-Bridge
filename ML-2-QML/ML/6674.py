"""QuantumClassifierModel – Classical implementation using PyTorch.

The class mirrors the original factory but now provides a full training
pipeline, evaluation helpers, and a flexible architecture.  It can be
easily swapped into existing ML workflows without changing the
interface.

Usage example:

    import torch
    from QuantumClassifierModel__gen526 import QuantumClassifierModel

    model = QuantumClassifierModel(num_features=20, depth=3)
    model.fit(X_train, y_train, epochs=50)
    preds = model.predict(X_test)
"""

from __future__ import annotations

from typing import Tuple, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class QuantumClassifierModel:
    """Deep feed‑forward classifier with batch‑norm and adjustable depth."""

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.device = device

        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def _build_model(self) -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            layers.extend(
                [
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.hidden_dim),
                ]
            )
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> None:
        """Train the network on (X_train, y_train)."""
        self.model.train()
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device).float()
                self.optimizer.zero_grad()
                logits = self.model(x).squeeze(-1)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * x.size(0)

            epoch_loss /= len(loader.dataset)
            if verbose and epoch % max(1, epochs // 10) == 0:
                if X_val is not None and y_val is not None:
                    val_acc = self.accuracy(X_val, y_val)
                    print(
                        f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - val acc: {val_acc:.4f}"
                    )
                else:
                    print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f}")

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return class predictions for X."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device)).squeeze(-1)
            probs = torch.sigmoid(logits)
            return (probs >= threshold).long()

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Compute classification accuracy."""
        preds = self.predict(X)
        return (preds == y.long()).float().mean().item()

    def get_parameters(self) -> list[float]:
        """Return flattened parameter vector."""
        return [p.detach().cpu().numpy().flatten() for p in self.model.parameters()]

    def __repr__(self) -> str:
        return (
            f"QuantumClassifierModel(num_features={self.num_features}, "
            f"depth={self.depth}, hidden_dim={self.hidden_dim})"
        )


__all__ = ["QuantumClassifierModel"]
