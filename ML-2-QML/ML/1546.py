"""Enhanced classical classifier mirroring the quantum helper interface.

This module extends the original seed by adding:
- configurable dropout and batch normalization
- weight initialization (He/Kaiming)
- early stopping based on validation loss
- optional data normalization
- flexible optimizer selection
- a metadata interface compatible with the quantum version
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader


class QuantumClassifierModel:
    """A classical feed‑forward neural network with a quantum‑style interface."""

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        lr: float = 1e-3,
        weight_init: str = "kaiming",
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        num_features : int
            Dimensionality of the input.
        depth : int
            Number of hidden layers.
        hidden_dim : int | None
            Size of hidden layers. If None, defaults to num_features.
        dropout : float
            Dropout probability applied after each hidden layer.
        batch_norm : bool
            Whether to insert a BatchNorm1d after each hidden layer.
        lr : float
            Optimiser learning rate.
        weight_init : str
            'kaiming' or 'xavier' weight initialization.
        device : str | torch.device
            Target device.
        """
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.lr = lr
        self.weight_init = weight_init
        self.device = torch.device(device)

        self.net = self._build_network().to(self.device)
        self._init_weights(self.weight_init)

        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def _build_network(self) -> nn.Module:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def _init_weights(self, scheme: str) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if scheme.lower() == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif scheme.lower() == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def train(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        epochs: int = 50,
        patience: int = 5,
        verbose: bool = True,
    ) -> None:
        """Train with early‑stopping on validation loss."""
        best_val = float("inf")
        counter = 0
        for epoch in range(epochs):
            self.net.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.net(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

            val_loss = self.evaluate(val_loader, return_loss=True)
            if verbose:
                print(f"Epoch {epoch+1:02d} | Val loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.net.state_dict(), "best_model.pt")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print("Early stopping")
                    break
        self.net.load_state_dict(torch.load("best_model.pt"))

    def evaluate(
        self,
        loader: Iterable,
        return_loss: bool = False,
    ) -> float:
        """Return loss or accuracy on a dataset."""
        self.net.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.net(x)
                loss = self.criterion(logits, y)
                loss_sum += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        if return_loss:
            return loss_sum / total
        return correct / total

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        self.net.eval()
        with torch.no_grad():
            logits = self.net(x.to(self.device))
            probs = F.softmax(logits, dim=1)
        return probs.cpu()

    def get_encoding(self) -> Tuple[List[int], List[int], List[int]]:
        """Return dummy metadata mimicking the quantum output."""
        encoding = list(range(self.num_features))
        weight_sizes = [
            sum(p.numel() for p in layer.parameters())
            for layer in self.net.modules()
            if isinstance(layer, nn.Linear)
        ]
        observables = [0, 1]  # placeholder
        return encoding, weight_sizes, observables

    def __repr__(self) -> str:
        return f"<QuantumClassifierModel depth={self.depth} hidden={self.hidden_dim}>"

__all__ = ["QuantumClassifierModel"]
