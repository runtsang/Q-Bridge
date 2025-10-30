"""QuantumClassifier: a classical feed‑forward network that mirrors the quantum interface.

The class exposes the same public API as the quantum counterpart: ``build`` returns a
PyTorch model, ``encoding`` gives the feature indices, ``weight_sizes`` reports the
number of trainable parameters per layer, and ``observables`` lists the output
class indices.  Dropout and batch‑normalisation layers are optional, allowing the
user to experiment with regularisation without changing the interface."""
from __future__ import annotations

from typing import Iterable, List, Tuple
import torch
import torch.nn as nn


class QuantumClassifier:
    """Classical feed‑forward neural network with optional regularisation."""

    def __init__(
        self,
        num_features: int,
        depth: int,
        dropout: float = 0.0,
        batch_norm: bool = False,
        hidden_size: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        num_features:
            Number of input features.
        depth:
            Number of hidden layers.
        dropout:
            Dropout probability applied after every hidden layer. 0.0 disables dropout.
        batch_norm:
            If True, a BatchNorm1d layer follows each linear layer.
        hidden_size:
            Size of hidden layers.  If ``None`` defaults to ``num_features``.
        """
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.hidden_size = hidden_size or num_features
        self.model, self.encoding, self.weight_sizes, self.observables = self.build()

    def build(
        self,
    ) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
        """Return the network, encoding indices, weight‑size metadata and output classes."""
        layers: List[nn.Module] = []
        in_dim = self.num_features
        encoding = list(range(self.num_features))
        weight_sizes: List[int] = []

        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.hidden_size)
            layers.append(linear)
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())

            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))

            in_dim = self.hidden_size

        head = nn.Linear(in_dim, 2)  # binary classification
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = [0, 1]
        return network, encoding, weight_sizes, observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)

    def __repr__(self) -> str:
        return f"<QuantumClassifier depth={self.depth} hidden={self.hidden_size} dropout={self.dropout} batch_norm={self.batch_norm}>"
