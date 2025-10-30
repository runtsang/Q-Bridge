"""Hybrid classical classifier with enhanced depth and regularisation."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """
    A classical feed‑forward classifier that mirrors the quantum helper's
    factory interface while providing additional training knobs such as
    residual connections, dropout, and a configurable hidden dimension.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        """
        Parameters
        ----------
        num_features : int
            Number of input features.
        depth : int
            Number of hidden layers.
        hidden_dim : int | None, optional
            Width of the hidden layers.  If None, defaults to ``num_features``.
        dropout : float, optional
            Dropout probability applied after each ReLU.  Defaults to 0.0.
        residual : bool, optional
            If True, adds a simple residual connection from the input to
            the final hidden layer.  Useful for mitigating vanishing gradients.
        """
        super().__init__()
        hidden_dim = hidden_dim or num_features

        layers: List[nn.Module] = []
        in_dim = num_features
        encoding: List[int] = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.extend([linear, nn.ReLU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = hidden_dim

        self.residual = nn.Linear(num_features, hidden_dim) if residual else None

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        self.network = nn.Sequential(*layers)
        self.encoding = encoding
        self.weight_sizes = weight_sizes
        self.observables = [0, 1]  # placeholder that matches the original API

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual is not None:
            x = self.network(x) + self.residual(x)
        else:
            x = self.network(x)
        return x

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """
        Factory that returns a tuple matching the original seed:
        (network, encoding, weight_sizes, observables).
        """
        model = QuantumClassifierModel(
            num_features, depth, hidden_dim, dropout, residual
        )
        return model.network, model.encoding, model.weight_sizes, model.observables


__all__ = ["QuantumClassifierModel"]
