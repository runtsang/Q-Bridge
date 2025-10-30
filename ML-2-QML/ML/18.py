"""Classical classifier factory with advanced training utilities."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QuantumClassifierModelGen:
    """A flexible feed‑forward classifier that mimics the quantum helper interface.

    The constructor builds a network with configurable depth, activation, and dropout.
    The :func:`build_classifier_circuit` factory returns the network together with
    encoding metadata, weight sizes and observables so that downstream code can
    introspect the model in the same shape as the quantum counterpart.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: Optional[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features
        self.activation = getattr(nn, activation.capitalize())()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.device = device

        # Build the network
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(self.activation)
            layers.append(self.dropout)
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers).to(self.device)

        # Metadata used by the factory interface
        self.encoding = list(range(num_features))
        self.weight_sizes = [
            sum(p.numel() for p in layer.parameters()) for layer in self.network
        ]
        self.observables = list(range(2))

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        hidden_dim: Optional[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Factory that returns a freshly‑constructed network and its metadata.

        The signature mirrors the quantum version so that users can swap implementations
        without changing downstream code.  The returned ``encoding`` and ``observables``
        are simple integer lists that emulate the qubit indices used in the quantum
        implementation.
        """
        dummy = QuantumClassifierModelGen(
            num_features,
            depth,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
        )
        return dummy.network, dummy.encoding, dummy.weight_sizes, dummy.observables

    # --------------------------------------------------------------------------- #
    # Training utilities – simple wrappers around PyTorch
    # --------------------------------------------------------------------------- #

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        optimizer_name: str = "adam",
    ) -> None:
        """Train the network on ``X, y`` using a classic optimizer.

        Parameters
        ----------
        X
            Input features of shape (N, num_features).
        y
            Integer labels of shape (N,).
        """
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32, device=self.device),
            torch.tensor(y, dtype=torch.long, device=self.device),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = {
            "adam": optim.Adam(self.network.parameters(), lr=lr),
            "sgd": optim.SGD(self.network.parameters(), lr=lr),
        }[optimizer_name.lower()]

        self.network.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.network(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions for ``X``."""
        self.network.eval()
        with torch.no_grad():
            logits = self.network(torch.tensor(X, dtype=torch.float32, device=self.device))
            return logits.argmax(dim=1).cpu().numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return cross‑entropy loss on a held‑out set."""
        self.network.eval()
        with torch.no_grad():
            logits = self.network(torch.tensor(X, dtype=torch.float32, device=self.device))
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, torch.tensor(y, dtype=torch.long, device=self.device)).item()


__all__ = ["QuantumClassifierModelGen"]
