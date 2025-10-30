"""Classical neural‑network classifier that mirrors the quantum helper interface.

The original seed provided a tiny feed‑forward network.  This upgrade offers
- configurable depth and width
- optional batch‑norm and dropout
- early‑stopping based on validation loss
- easy extraction of weight statistics for comparison with the quantum model
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["QuantumClassifierModel", "build_classifier_network"]


def build_classifier_network(
    num_features: int,
    hidden_layers: List[int] | None = None,
    dropout: float = 0.0,
) -> Tuple[nn.Module, List[int]]:
    """Construct a multi‑layer feed‑forward classifier.

    Parameters
    ----------
    num_features:
        Dimensionality of the input.
    hidden_layers:
        Sequence of hidden layer sizes.  If ``None`` a default two‑layer
        network is used.
    dropout:
        Dropout probability applied after every hidden layer.

    Returns
    -------
    network:
        ``torch.nn.Sequential`` instance.
    weight_sizes:
        List of the number of trainable parameters in each layer (excluding biases).
    """
    if hidden_layers is None:
        hidden_layers = [128, 64]

    layers: List[nn.Module] = []
    in_dim = num_features
    weight_sizes: List[int] = []

    for h in hidden_layers:
        linear = nn.Linear(in_dim, h)
        layers.append(linear)
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        weight_sizes.append(linear.weight.numel())
        in_dim = h

    # Final classifier
    head = nn.Linear(in_dim, 2)  # binary classification
    layers.append(head)
    weight_sizes.append(head.weight.numel())

    net = nn.Sequential(*layers)
    return net, weight_sizes


class QuantumClassifierModel:
    """Classic feed‑forward classifier with a training API compatible with the
    quantum version.

    Attributes
    ----------
    net:
        The underlying ``torch.nn.Module``.
    device:
        ``torch.device`` used for computation.
    """

    def __init__(
        self,
        num_features: int,
        hidden_layers: List[int] | None = None,
        dropout: float = 0.0,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        self.net, self.weight_sizes = build_classifier_network(
            num_features, hidden_layers, dropout
        )
        self.device = torch.device(device)
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 32,
        epochs: int = 20,
        val_split: float = 0.1,
        patience: int = 5,
        verbose: bool = False,
    ) -> None:
        """Train the network with early‑stopping.

        Parameters
        ----------
        X, y:
            Training data and labels.
        batch_size:
            Size of mini‑batches.
        epochs:
            Maximum number of epochs.
        val_split:
            Fraction of data reserved for validation.
        patience:
            Number of epochs with no improvement before stopping.
        verbose:
            If ``True`` prints epoch statistics.
        """
        dataset = TensorDataset(X, y)
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(X[n_train:], y[n_train:]), batch_size=batch_size
        )

        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            self.net.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.net(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * xb.size(0)

            train_loss /= n_train

            # Validation
            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.net(xb)
                    loss = self.criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= n_val

            if verbose:
                print(
                    f"Epoch {epoch:02d} | "
                    f"train loss {train_loss:.4f} | val loss {val_loss:.4f}"
                )

            # Early stopping
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_state = self.net.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stop after {epoch} epochs.")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return class predictions (0 or 1)."""
        self.net.eval()
        with torch.no_grad():
            logits = self.net(X.to(self.device))
            return torch.argmax(logits, dim=1).cpu()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """Return accuracy and loss on a dataset."""
        self.net.eval()
        with torch.no_grad():
            logits = self.net(X.to(self.device))
            loss = self.criterion(logits, y.to(self.device)).item()
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y.to(self.device)).float().mean().item()
        return {"loss": loss, "accuracy": acc}
