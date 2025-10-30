"""Enhanced classical classifier for hybrid experiments.

The model implements an attention‑weighted MLP with automatic depth tuning
and early‑stopping support.  It follows the same public API as the
seed (`build_classifier_circuit`) so that the quantum and classical
components can be swapped interchangeably in downstream pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List, Optional

__all__ = ["build_classifier_circuit", "QuantumClassifierModel"]


class QuantumClassifierModel(nn.Module):
    """Attention‑based MLP that mirrors the quantum interface.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input data.
    hidden_size : int, default 64
        Width of the hidden layers.
    depth : int, default 3
        Number of hidden layers.
    dropout : float, default 0.1
        Drop‑out probability applied after each hidden layer.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.depth = depth

        # Attention layer that learns a feature‑wise weight
        self.attention = nn.Linear(num_features, num_features, bias=False)

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class scores."""
        # Feature‑wise attention
        att = torch.softmax(self.attention(x), dim=1)
        x = x * att
        x = self.body(x)
        logits = self.head(x)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the predicted class label."""
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            return torch.argmax(probs, dim=1)

    def evaluate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and probabilities for a batch."""
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def train_loop(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 5,
        device: Optional[torch.device] = None,
    ) -> None:
        """Simple training loop with early stopping.

        Parameters
        ----------
        train_loader : Iterable[Tuple[Tensor, Tensor]]
            DataLoader yielding (x, y) pairs.
        val_loader : Iterable[Tuple[Tensor, Tensor]]
            Validation DataLoader.
        epochs : int
            Maximum number of epochs.
        lr : float
            Learning rate.
        weight_decay : float
            L2 regularisation.
        patience : int
            Number of epochs with no improvement before stopping.
        device : torch.device, optional
            Device to run the model on. If None, uses 'cpu'.
        """
        device = device or torch.device("cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = self(xb)
                    val_loss += criterion(logits, yb).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)


def build_classifier_circuit(
    num_features: int,
    depth: int = 3,
    hidden_size: int = 64,
    dropout: float = 0.1,
) -> Tuple[QuantumClassifierModel, List[int], List[int], List[int]]:
    """Factory that mirrors the quantum helper interface.

    Returns
    -------
    model : QuantumClassifierModel
        The constructed attention‑MLP.
    encoding : List[int]
        Feature indices used for encoding (here all features).
    weight_sizes : List[int]
        Number of trainable parameters per layer.
    observables : List[int]
        Dummy observable list for API compatibility.
    """
    model = QuantumClassifierModel(
        num_features=num_features,
        hidden_size=hidden_size,
        depth=depth,
        dropout=dropout,
    )
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = list(range(2))
    return model, encoding, weight_sizes, observables
