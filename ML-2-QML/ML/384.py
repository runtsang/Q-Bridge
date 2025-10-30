"""Hybrid quantum classifier: classical side.

Provides a PyTorch model that can optionally incorporate a quantum feature extractor.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class QuantumClassifierModel(nn.Module):
    """
    PyTorch classifier that mirrors the quantum helper interface but adds
    dropout, batch‑norm and optional quantum feature extraction.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_sizes: List[int] | None = None,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        quantum_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """
        Parameters
        ----------
        num_features
            Number of input features (classical).
        depth
            Number of hidden layers if ``hidden_sizes`` is None.
        hidden_sizes
            Sequence of hidden layer sizes. If None, a symmetric network
            with ``depth`` layers of size ``num_features`` is created.
        dropout
            Dropout probability applied after each hidden layer.
        use_batchnorm
            Whether to insert BatchNorm after each hidden layer.
        quantum_extractor
            Callable that maps a classical tensor to a quantum‑encoded tensor.
            Should accept a tensor of shape (batch, num_features) and return
            a tensor of shape (batch, feature_dim). If None, no quantum
            preprocessing is performed.
        """
        super().__init__()
        self.quantum_extractor = quantum_extractor

        if hidden_sizes is None:
            hidden_sizes = [num_features] * depth

        layers: List[nn.Module] = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        If ``quantum_extractor`` is set, it is applied before the feature
        extractor. This mirrors the quantum circuit's output dimensions
        being fed into a classical head.
        """
        if self.quantum_extractor is not None:
            x = self.quantum_extractor(x)
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stop_patience: int | None = None,
    ) -> None:
        """
        Train the model using Adam.

        Parameters
        ----------
        X, y
            Training data and labels.
        epochs
            Maximum number of epochs.
        batch_size
            Mini‑batch size.
        lr
            Learning rate.
        weight_decay
            L2 regularisation.
        early_stop_patience
            Number of epochs with no improvement before stopping.
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # simple validation on training set for demonstration
            self.eval()
            with torch.no_grad():
                logits = self(X)
                val_loss = criterion(logits, y).item()

            if early_stop_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return class indices.
        """
        self.eval()
        with torch.no_grad():
            logits = self(X)
        return torch.argmax(logits, dim=1)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Return accuracy on the provided data.
        """
        preds = self.predict(X)
        return (preds == y).float().mean().item()


__all__ = ["QuantumClassifierModel"]
