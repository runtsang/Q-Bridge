"""Classical classifier with extended architecture and training utilities.

The implementation builds on the original feed‑forward factory by adding optional
dropout, batch‑normalisation, and a small training loop that can be used directly
in research pipelines.  The public API mirrors the seed implementation: a
function ``build_classifier_circuit`` that returns a network and metadata, and a
``QuantumClassifier`` class that exposes a training API.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier and metadata similar to the quantum variant.

    Parameters
    ----------
    num_features : int
        Number of input features / qubits.
    depth : int
        Number of hidden layers.
    hidden_dim : int | None, default=None
        Size of hidden layers.  If ``None`` the hidden layers match ``num_features``.
    dropout : float, default=0.0
        Dropout probability applied after each hidden layer.

    Returns
    -------
    network : nn.Module
        The constructed network.
    encoding : list[int]
        Indices of input features.
    weight_sizes : list[int]
        Number of trainable parameters per layer.
    observables : list[int]
        Dummy observable indices (mirroring the quantum implementation).
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    hidden_dim = hidden_dim or num_features
    for _ in range(depth):
        linear = nn.Linear(in_dim, hidden_dim)
        layers.append(linear)
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = hidden_dim

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class QuantumClassifier:
    """Wrapper around the classical feed‑forward network that exposes a training API."""

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth, hidden_dim, dropout
        )
        self.network.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 20,
        verbose: bool = False,
    ) -> list[float]:
        """Train the network for a fixed number of epochs.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader yielding ``(X, y)`` batches.
        epochs : int, default=20
            Number of training epochs.
        verbose : bool, default=False
            Whether to print epoch‑wise loss.

        Returns
        -------
        history : list[float]
            List of average training loss per epoch.
        """
        history: list[float] = []
        for epoch in range(epochs):
            self.network.train()
            epoch_loss = 0.0
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.network(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * X.size(0)
            epoch_loss /= len(dataloader.dataset)
            history.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss:.4f}")
        return history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for the given inputs.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape ``(N, num_features)``.

        Returns
        -------
        probs : torch.Tensor
            Probabilities of shape ``(N, 2)``.
        """
        self.network.eval()
        with torch.no_grad():
            logits = self.network(X.to(self.device))
            probs = torch.softmax(logits, dim=-1)
        return probs


__all__ = ["build_classifier_circuit", "QuantumClassifier"]
