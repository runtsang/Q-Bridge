"""Classical multi‑class MLP that mirrors the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QuantumClassifierModel:
    """
    Classical MLP classifier mimicking the quantum API.

    The constructor builds a feed‑forward network with a user‑specified depth.
    The static :meth:`build_classifier_circuit` returns the underlying ``nn.Module``
    and metadata identical to the quantum counterpart, enabling easy substitution
    in hybrid workflows.
    """

    def __init__(self, num_features: int, depth: int = 2, hidden_dim: int | None = None):
        hidden_dim = hidden_dim or num_features
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
        self.encoding: List[int] = list(range(num_features))
        self.weight_sizes: List[int] = [p.numel() for p in self.net.parameters()]
        self.observables: List[int] = list(range(2))

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Factory that reproduces the signature of the quantum helper.
        Returns the ``nn.Module`` and the three metadata lists.
        """
        model = QuantumClassifierModel(num_features, depth)
        return model.net, model.encoding, model.weight_sizes, model.observables

    def train(self, X: torch.Tensor, y: torch.Tensor,
              lr: float = 1e-3, epochs: int = 200, batch_size: int = 32,
              verbose: bool = False) -> None:
        """
        Standard supervised training loop for a cross‑entropy loss.

        Parameters
        ----------
        X : torch.Tensor
            Input features (N × d).
        y : torch.Tensor
            Binary labels (0/1) of shape (N,).
        """
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if verbose and (epoch + 1) % 20 == 0:
                print(f"[ML] Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(loader):.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return class predictions (0 or 1).
        """
        with torch.no_grad():
            logits = self.net(X)
            return logits.argmax(dim=1)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Accuracy on the provided dataset.
        """
        preds = self.predict(X)
        return (preds == y).float().mean().item()
