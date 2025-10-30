"""Hybrid classifier class with an extendable classical neural network."""
from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """
    Classical feed‑forward network that mirrors the interface of its quantum counterpart.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    residual : bool, default False
        If ``True``, each hidden layer is wrapped in a residual connection.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.
    batchnorm : bool, default False
        If ``True``, a batch‑norm layer follows each hidden layer.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        residual: bool = False,
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        self.num_features = num_features
        self.depth = depth
        self.residual = residual
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.network, self.weight_sizes = self._build_network()
        self.encoding = list(range(num_features))
        self.observables = [0, 1]  # placeholder for compatibility

    def _build_network(self) -> tuple[nn.Sequential, List[int]]:
        """Constructs the feed‑forward network with optional residuals."""
        layers: List[nn.Module] = []
        in_dim = self.num_features

        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.num_features)
            layers.append(linear)

            if self.batchnorm:
                layers.append(nn.BatchNorm1d(self.num_features))
            layers.append(nn.ReLU())

            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))

            # Residual shortcut
            if self.residual:
                # Identity mapping if input and output dimensionality differ
                if in_dim!= self.num_features:
                    layers.append(nn.Linear(in_dim, self.num_features))
                layers.append(nn.ReLU())

            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = self.num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        return network, self.weight_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

    def get_encoding(self) -> List[int]:
        """Return the feature indices used for encoding."""
        return self.encoding

    def get_weight_sizes(self) -> List[int]:
        """Return the list of weight + bias counts per layer."""
        return self.weight_sizes

__all__ = ["QuantumClassifierModel"]
