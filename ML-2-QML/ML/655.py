"""Classical classifier factory mirroring the quantum helper interface.

The class encapsulates a feed‑forward network with optional dropout,
batch‑normalisation and a simple training pipeline.  It preserves the
original seed signature by exposing a static `build_classifier_circuit`
method that returns the network, encoding, weight sizes and observables.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class QuantumClassifierModel:
    """
    Classical feed‑forward classifier that mimics the quantum circuit factory.

    Parameters
    ----------
    num_features : int
        Number of input features (analogous to qubits).
    depth : int
        Number of hidden layers; each layer has `num_features` units.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    batch_norm : bool, optional
        Whether to insert a BatchNorm1d layer after each hidden layer.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.batch_norm = batch_norm

        layers: List[nn.Module] = []
        in_dim = num_features

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features

        self.head = nn.Linear(in_dim, 2)
        layers.append(self.head)
        self.network = nn.Sequential(*layers)

        # Mirrors the seed outputs
        self.encoding = list(range(num_features))
        self.weight_sizes = [
            m.weight.numel() + m.bias.numel()
            for m in self.network.modules()
            if isinstance(m, nn.Linear)
        ]
        self.observables = list(range(2))

    # ------------------------------------------------------------------
    # Forward and utilities
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""
        return self.network(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict class labels for input tensor `x`."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).long()

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        """Mini‑batch training loop."""
        self.network.to(device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            self.network.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self.network(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_encoding(self) -> List[int]:
        """Return the encoding vector (input feature indices)."""
        return self.encoding

    def get_weight_sizes(self) -> List[int]:
        """Return a list of parameter counts per linear layer."""
        return self.weight_sizes

    def get_observables(self) -> List[int]:
        """Return a list of observable indices (here just 0 and 1)."""
        return self.observables

    # ------------------------------------------------------------------
    # Convenience wrapper matching the seed function signature
    # ------------------------------------------------------------------
    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Static factory that returns the same tuple as the original seed
        function but with a richer network.
        """
        model = QuantumClassifierModel(num_features, depth)
        return model.network, model.get_encoding(), model.get_weight_sizes(), model.get_observables()
