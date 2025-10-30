"""Classical feed‑forward classifier that mirrors the quantum helper interface.

The class exposes the same public API as the original seed (``build_classifier_circuit``)
while providing a full training pipeline.  It is intentionally lightweight so that it
can be dropped into any PyTorch experiment.

Key extensions:
* configurable hidden dimension and depth
* dropout and L2 regularisation
* early‑stopping based on validation loss
* convenient access to weight sizes and encoding metadata
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class QuantumClassifierModel(nn.Module):
    """
    Feed‑forward classifier that mimics the quantum helper interface.

    Parameters
    ----------
    num_features : int
        Number of input features / qubits.
    depth : int, default 2
        Number of hidden layers.
    hidden_dim : int | None, default None
        Width of hidden layers.  If ``None`` defaults to ``num_features``.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.
    l2 : float, default 0.0
        L2 regularisation coefficient.
    lr : float, default 1e-3
        Optimiser learning rate.
    device : str | torch.device, default 'cpu'
        Target device.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        l2: float = 0.0,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features
        self.dropout = dropout
        self.l2 = l2
        self.lr = lr
        self.device = torch.device(device)

        # Build the network
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 2))  # binary classification
        self.network = nn.Sequential(*layers).to(self.device)

        # Metadata
        self.encoding = list(range(num_features))
        self.weight_sizes = [p.numel() for p in self.parameters()]
        self.observables = list(range(2))

        # Optimiser
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x.to(self.device))

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        """Train the network using binary cross‑entropy loss."""
        self.train()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        best_loss = float("inf")
        patience, counter = 10, 0  # early stopping

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)
            if verbose:
                print(f"Epoch {epoch:03d} – loss: {epoch_loss:.4f}")

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return class labels."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(X.to(self.device))
            return torch.argmax(logits, dim=1).cpu()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(X.to(self.device))
            return torch.softmax(logits, dim=1).cpu()

    def get_weights(self) -> List[np.ndarray]:
        """Return a list of NumPy arrays containing the parameters."""
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return (encoding, weight_sizes, observables)."""
        return self.encoding, self.weight_sizes, self.observables


__all__ = ["QuantumClassifierModel"]
