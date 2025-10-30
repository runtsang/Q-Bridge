"""Enhanced classical classifier factory with richer architecture and introspection."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable

import torch
import torch.nn as nn

class QuantumClassifierModel(nn.Module):
    """
    A fully‑connected neural network that mirrors the interface of the quantum
    helper but offers richer architectural options.

    Parameters
    ----------
    num_features : int
        Input feature dimension.
    depth : int, default 2
        Number of hidden layers.
    hidden_dim : int | None, default None
        Size of each hidden layer; defaults to ``num_features``.
    dropout : float, default 0.0
        Drop‑out probability applied after each hidden layer.
    activation : Callable[[nn.Module], nn.Module], default nn.ReLU
        Activation function class.
    batchnorm : bool, default False
        Apply a BatchNorm1d after each linear layer.
    output_dim : int, default 2
        Size of the output layer.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        activation: Callable[[nn.Module], nn.Module] = nn.ReLU,
        batchnorm: bool = False,
        output_dim: int = 2,
    ):
        super().__init__()
        hidden_dim = hidden_dim or num_features
        layers: List[nn.Module] = []
        in_dim = num_features

        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(linear)
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        head = nn.Linear(in_dim, output_dim)
        layers.append(head)
        self.network = nn.Sequential(*layers)

        # Metadata for compatibility with QML interface
        self.encoding = list(range(num_features))
        self.weight_sizes = [p.numel() for p in self.parameters()]
        self.observables = [f"obs_{i}" for i in range(output_dim)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_dict(self) -> dict:
        """Export a serialisable representation of the model."""
        return {
            "num_features": len(self.encoding),
            "depth": len(self.network) // 2,
            "hidden_dim": self.network[0].out_features,
            "output_dim": self.network[-1].out_features,
            "weight_sizes": self.weight_sizes,
            "observables": self.observables,
        }

__all__ = ["QuantumClassifierModel"]
