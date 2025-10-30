"""Classical classifier mirroring quantum helper interface with enhanced features.

The class builds a configurable feed‑forward network, supports dropout,
early‑stopping, and exposes a minimal training API.  It also returns
metadata (encoding, weight_sizes, observables) to keep parity with the
quantum helper.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]


class QuantumClassifierModel(nn.Module):
    """Feed‑forward classifier with optional dropout and early stopping."""

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features
        self.dropout = dropout
        self.device = device

        self.network = self._build_network().to(device)

    def _build_network(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.to(self.device))

    def train_loop(
        self,
        train_loader,
        epochs: int,
        lr: float,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        early_stop_patience: int = 5,
    ) -> None:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = float("inf")
        patience = early_stop_patience

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for X, y in train_loader:
                optimizer.zero_grad()
                logits = self.forward(X)
                loss = loss_fn(logits, y.to(self.device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = early_stop_patience
            else:
                patience -= 1
                if patience <= 0:
                    break

    def evaluate(self, loader) -> Tuple[float, int]:
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in loader:
                logits = self.forward(X)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y.to(self.device)).sum().item()
                total += y.size(0)
        return correct / total, total

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            return torch.argmax(logits, dim=1)


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Return a tuple that mirrors the quantum helper signature.

    Parameters
    ----------
    num_features : int
        Number of input features / qubits.
    depth : int
        Number of hidden layers.
    hidden_dim : int, optional
        Size of hidden layers; defaults to ``num_features``.
    dropout : float, optional
        Dropout probability.

    Returns
    -------
    Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]
        The network, an encoding mask, a list of weight sizes, and a set
        of observable indices.
    """
    model = QuantumClassifierModel(
        num_features=num_features,
        depth=depth,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = list(range(2))
    return model, encoding, weight_sizes, observables
