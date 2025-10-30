"""Enhanced classical classifier with PyTorch, early stopping, and cross‑validation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from typing import Iterable, Tuple, List

class QuantumClassifier:
    """
    A classical feed‑forward classifier that mirrors the quantum interface.
    Supports training with early stopping, learning‑rate scheduling, and
    k‑fold cross‑validation.  The API is intentionally compatible with the
    quantum counterpart so that experiments can be swapped at runtime.
    """

    def __init__(self,
                 num_features: int,
                 depth: int = 3,
                 hidden_dim: int = 64,
                 lr: float = 1e-3,
                 batch_size: int = 128,
                 epochs: int = 200,
                 patience: int = 10,
                 seed: int | None = None):
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.model = self._build_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5, verbose=False
        )
        self.criterion = nn.CrossEntropyLoss()
        self.best_state = None
        self.best_loss = float("inf")

    def _build_network(self) -> nn.Module:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray, *,
            val_split: float = 0.1, shuffle: bool = True) -> None:
        """
        Train the network using early stopping on the validation loss.
        """
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.long))
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_loader = DataLoader(dataset[:n_train], batch_size=self.batch_size,
                                  shuffle=shuffle)
        val_loader = DataLoader(dataset[n_train:], batch_size=self.batch_size,
                                shuffle=False)

        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

            val_loss = self._evaluate(val_loader)
            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                losses.append(loss.item())
        return np.mean(losses)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            probs = nn.functional.softmax(logits, dim=1)
        return probs.numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict(X)
        preds = np.argmax(probs, axis=1)
        return np.mean(preds == y)

    def cross_validate(self, X: np.ndarray, y: np.ndarray, k: int = 5) -> List[float]:
        """
        Return a list of accuracy scores for each fold.
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        scores = []
        for train_idx, val_idx in kf.split(X):
            self.fit(X[train_idx], y[train_idx], val_split=0.0)
            scores.append(self.score(X[val_idx], y[val_idx]))
        return scores

__all__ = ["QuantumClassifier"]
