"""Enhanced classical classifier mirroring the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


def build_classifier_circuit(
    num_features: int,
    depth: int = 2,
    hidden_activation: nn.Module = nn.ReLU(),
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward network with optional dropout and batch‑norm.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature space.
    depth : int, default 2
        Number of hidden layers.
    hidden_activation : nn.Module, default nn.ReLU()
        Activation function applied after each hidden layer.
    dropout : float, default 0.0
        Drop‑out probability. 0 disables dropout.
    batch_norm : bool, default False
        Whether to insert a BatchNorm1d layer after each hidden linear layer.

    Returns
    -------
    network : nn.Sequential
        The constructed classifier.
    encoding : list[int]
        Feature indices used by the original quantum interface.
    weight_sizes : list[int]
        Total number of trainable parameters per layer (weights + biases).
    observables : list[int]
        Dummy observable list mirroring the quantum API.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())

        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features))

        layers.append(hidden_activation)

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        in_dim = num_features

    # Output head: binary classification
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # class indices for compatibility
    return network, encoding, weight_sizes, observables


class QuantumClassifierModel:
    """
    Classical analogue of the quantum classifier.

    The class exposes the same public methods as the Qiskit version:
    * ``predict`` – returns logits
    * ``predict_proba`` – returns softmax probabilities
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        *,
        hidden_activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features,
            depth,
            hidden_activation=hidden_activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.num_features = num_features

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax."""
        logits = self.predict(x)
        return torch.softmax(logits, dim=1)
