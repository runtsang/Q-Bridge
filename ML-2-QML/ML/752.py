"""QuantumClassifierModel: A flexible classical neural network wrapper.

This class extends the original feed‑forward factory by allowing:
* arbitrary hidden layer sizes and depths
* dropout regularisation
* early‑stopping during training
* metadata (encoding, weight sizes, observables) that mirrors the quantum side
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class QuantumClassifierModel:
    """Classical feed‑forward classifier with metadata compatible with the quantum variant."""
    def __init__(
        self,
        num_features: int,
        hidden_sizes: List[int] | None = None,
        depth: int = 2,
        dropout: float = 0.0,
        device: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        num_features: int
            Number of input features.
        hidden_sizes: list[int] | None
            Sizes of hidden layers. If None, ``depth`` fully connected layers of size ``num_features`` are used.
        depth: int
            Number of hidden layers when ``hidden_sizes`` is None.
        dropout: float
            Dropout probability applied after each hidden layer.
        device: str | None
            Torch device; defaults to CUDA if available.
        """
        self.num_features = num_features
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hidden_sizes = hidden_sizes or [num_features] * depth

        layers: List[nn.Module] = []
        in_dim = num_features
        self.encoding = list(range(num_features))
        self.weight_sizes: List[int] = []

        for h in hidden_sizes:
            lin = nn.Linear(in_dim, h)
            layers.append(lin)
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            self.weight_sizes.append(lin.weight.numel() + lin.bias.numel())
            in_dim = h

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())

        self.network = nn.Sequential(*layers).to(self.device)
        self.observables = [0, 1]  # class indices

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        early_stopping: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Train the network using Adam and cross‑entropy loss."""
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.network.parameters(), lr=lr)

        best_loss = float("inf")
        patience = 0

        for epoch in range(1, epochs + 1):
            self.network.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.network(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)
            if verbose:
                print(f"Epoch {epoch:3d} loss={epoch_loss:.4f}")

            if early_stopping is not None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= early_stopping:
                    if verbose:
                        print(f"Early stopping after {epoch} epochs.")
                    break

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return class predictions (0 or 1)."""
        self.network.eval()
        with torch.no_grad():
            logits = self.network(X.to(self.device))
            return torch.argmax(logits, dim=1).cpu()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Return accuracy on the provided data."""
        preds = self.predict(X)
        return (preds == y).float().mean().item()

    def get_metadata(self) -> Tuple[Iterable[int], Iterable[int], List[int]]:
        """Return encoding, weight_sizes, and observables."""
        return self.encoding, self.weight_sizes, self.observables

__all__ = ["QuantumClassifierModel"]
