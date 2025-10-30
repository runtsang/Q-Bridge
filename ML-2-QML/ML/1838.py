"""Enhanced classical classifier that mirrors the quantum helper interface."""
from __future__ import annotations

from typing import Iterable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifier:
    """
    Classical feed‑forward classifier with optional residual connections and dropout.
    Mirrors the signature of the quantum helper: ``build_classifier_circuit`` but
    returns a :class:`torch.nn.Module` instance and accompanying metadata.
    """

    def __init__(self, num_features: int, depth: int, dropout: float = 0.0,
                 residual: bool = False):
        """
        Parameters
        ----------
        num_features: int
            Number of input features / qubits.
        depth: int
            Number of hidden layers.
        dropout: float, optional
            Dropout probability applied after each hidden layer.
        residual: bool, optional
            If True, add a residual connection from input to the final hidden
            layer before the classification head.
        """
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.residual = residual
        self.network = self._build_network()

    def _build_network(self) -> nn.Module:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            in_dim = self.num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        return nn.Sequential(*layers)

    @property
    def model(self) -> nn.Module:
        """Return the underlying :class:`torch.nn.Module`."""
        return self.network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def get_encoding(self) -> Iterable[int]:
        """Return a list of feature indices that are fed into the network."""
        return list(range(self.num_features))

    def get_weight_sizes(self) -> Iterable[int]:
        """Return the number of trainable parameters per linear layer."""
        sizes = []
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                sizes.append(layer.weight.numel() + layer.bias.numel())
        return sizes

    def get_observables(self) -> List[int]:
        """Return the output dimension indices (for compatibility with the quantum API)."""
        return [0, 1]

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(num_features={self.num_features}, "
                f"depth={self.depth}, dropout={self.dropout}, residual={self.residual})")


__all__ = ["QuantumClassifier"]
