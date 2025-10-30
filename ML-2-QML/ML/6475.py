"""
Classical counterpart to the quantum UnifiedClassifier.
Provides a deep ReLU network with a tunable fully‑connected layer that mimics quantum expectation values.
"""

from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import torch
from torch import nn

class FCL(nn.Module):
    """
    Simple fully‑connected layer that returns a scalar expectation value.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

class UnifiedClassifier:
    """
    Classical feed‑forward classifier with a tunable fully‑connected layer.
    """
    def __init__(self, num_features: int, depth: int = 2, num_classes: int = 2) -> None:
        self.num_features = num_features
        self.depth = depth
        self.num_classes = num_classes
        self.network = self._build_network()
        self.fc_layer = FCL(num_features)

    def _build_network(self) -> nn.Sequential:
        layers = []
        in_dim = self.num_features
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.num_features))
            layers.append(nn.ReLU())
            in_dim = self.num_features
        layers.append(nn.Linear(in_dim, self.num_classes))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, lr: float = 0.01) -> None:
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.long))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.network(X_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            logits = self.network(X_tensor)
            return torch.argmax(logits, dim=1).numpy()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimics the quantum `run` interface by delegating to the fully‑connected layer.
        """
        return self.fc_layer.run(thetas)

__all__ = ["UnifiedClassifier", "FCL"]
