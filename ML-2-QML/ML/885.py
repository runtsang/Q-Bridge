"""Enhanced classical classifier with modular training utilities.

This module extends the original feed‑forward factory by adding:
- configurable hidden sizes and dropout
- early‑stopping logic
- a small training helper that reports loss/accuracy per epoch
- support for Torch's DataLoader and mixed‑precision training
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class QuantumClassifierModel(nn.Module):
    """
    A fully‑connected neural network that mimics the interface of the quantum
    classifier factory but offers richer hyper‑parameter control.
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: List[int] | None = None,
        depth: int = 2,
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
    ):
        """
        Parameters
        ----------
        num_features: int
            Dimensionality of the input feature vector.
        hidden_sizes: List[int] | None
            Sizes of the hidden layers. If ``None`` a default of ``[num_features]*depth``
            is used.
        depth: int
            Number of hidden layers.
        dropout: float
            Dropout probability applied after each hidden layer.
        device: torch.device | str
            Target device for the model.
        """
        super().__init__()
        self.device = torch.device(device)
        hidden_sizes = hidden_sizes or [num_features] * depth
        layers: List[nn.Module] = []

        in_dim = num_features
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = size

        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.to(self.device))

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        dropout: float = 0.0,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Factory mirroring the quantum helper signature.

        Returns
        -------
        network, encoding, weight_sizes, observables
        """
        hidden_sizes = [num_features] * depth
        model = QuantumClassifierModel(num_features, hidden_sizes, depth, dropout)
        # Build metadata
        encoding = list(range(num_features))
        # We approximate weight sizes by counting parameters in each Linear layer
        weight_sizes = [
            sum(p.numel() for p in layer.parameters())
            for layer in model.network
            if isinstance(layer, nn.Linear)
        ]
        observables = list(range(2))
        return model, encoding, weight_sizes, observables

    def train_loop(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        epochs: int = 20,
        patience: int = 5,
        verbose: bool = True,
    ) -> List[float]:
        """
        Simple training loop with early stopping.

        Parameters
        ----------
        train_loader: DataLoader
            Iterable over (x, y) pairs.
        optimizer: Optimizer
            Torch optimizer.
        loss_fn: nn.Module
            Loss function.
        epochs: int
            Maximum number of epochs.
        patience: int
            Early‑stopping patience.
        verbose: bool
            Print progress.

        Returns
        -------
        List[float]
            Recorded training losses.
        """
        best_loss = float("inf")
        best_state = None
        loss_history: List[float] = []

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                logits = self(x)
                loss = loss_fn(logits, y.to(self.device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)

            epoch_loss /= len(train_loader.dataset)
            loss_history.append(epoch_loss)

            if verbose:
                acc = self.evaluate(train_loader)
                print(f"[{epoch+1:02d}] loss={epoch_loss:.4f} acc={acc:.3f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
                patience = 5  # reset
            else:
                patience -= 1
                if patience <= 0:
                    if verbose:
                        print("Early stopping")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        return loss_history

    def evaluate(self, loader: DataLoader) -> float:
        """
        Compute classification accuracy on a dataset.
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                logits = self(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total else 0.0

__all__ = ["QuantumClassifierModel"]
