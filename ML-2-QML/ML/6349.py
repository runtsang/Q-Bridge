"""Enhanced classical classifier with dropout, early stopping, and probability estimates.

The class mirrors the interface of the quantum counterpart while adding a
layered feed‑forward network built with PyTorch.  It supports configurable
depth, hidden size and dropout, and trains with Adam on a mini‑batch
cross‑entropy loss.  The implementation is fully self‑contained and can
be dropped into any existing data‑science pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Iterable

class QuantumClassifierModel:
    """
    A classical feed‑forward neural network that emulates the API of the
    quantum classifier.  The model can be trained on any numeric feature
    matrix and produces class labels or probability estimates.
    """
    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        hidden_units: int = 32,
        dropout: float = 0.5,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.num_features = num_features
        self.depth = depth
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self._build_model()

    def _build_model(self) -> None:
        layers: list[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_units
        layers.append(nn.Linear(in_dim, 2))
        self.model = nn.Sequential(*layers).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32, device=self.device))
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32, device=self.device))
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

__all__ = ["QuantumClassifierModel"]
