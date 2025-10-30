"""QuantumClassifierModel: Classical PyTorch implementation mirroring the quantum helper interface."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterable, Tuple, List, Optional
from torch.utils.data import DataLoader, TensorDataset, random_split

class QuantumClassifierModel:
    """A flexible PyTorch classifier that mimics the quantum helper interface.

    The class exposes the same constructor arguments as the original seed
    (``num_features`` and ``depth``) but internally builds a fully connected
    network with a configurable hidden dimensionality.  The public API
    (``fit``, ``predict`` and ``evaluate``) is intentionally similar to the
    Qiskit implementation so that the two modules can be swapped in a
    downstream experiment pipeline.
    """

    def __init__(self,
                 num_features: int,
                 depth: int,
                 hidden_dim: int = 32,
                 output_dim: int = 2,
                 lr: float = 1e-3,
                 epochs: int = 200,
                 batch_size: int = 64,
                 device: Optional[str] = None) -> None:
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Build network
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers).to(self.device)

        # Metadata
        self.encoder = list(range(num_features))
        self.weight_sizes = [p.numel() for p in self.model.parameters()]
        self.observables = list(range(output_dim))

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs

    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            val_split: float = 0.1,
            early_stopping_patience: int = 10) -> None:
        """Train the model with optional early stopping."""
        dataset = TensorDataset(X, y)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        best_val_loss = float("inf")
        patience = 0

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

            val_loss = self._evaluate_loader(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    break

    def _evaluate_loader(self, loader: DataLoader) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                losses.append(self.criterion(logits, yb).item())
        return sum(losses) / len(losses)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return class indices."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            return torch.argmax(logits, dim=1).cpu()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Return accuracy and cross‑entropy loss."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            loss = self.criterion(logits, y.to(self.device)).item()
            preds = torch.argmax(logits, dim=1)
            acc = (preds.cpu() == y).float().mean().item()
        return acc, loss

    def get_metadata(self) -> Tuple[List[int], List[int], List[str]]:
        """Return encoder, weight sizes and observables."""
        return self.encoder, self.weight_sizes, self.observables
