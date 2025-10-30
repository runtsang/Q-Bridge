"""Enhanced classical classifier mirroring the quantum interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifier(nn.Module):
    """
    Feed‑forward classifier that emulates the quantum circuit layout.

    Parameters
    ----------
    input_dim : int
        Number of input features / qubits.
    depth : int
        Number of hidden layers.
    hidden_dim : int, optional
        Width of each hidden layer (default: input_dim).
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    batch_norm : bool, optional
        Whether to insert BatchNorm after every hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        depth: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        layers: List[nn.Module] = []

        in_dim = input_dim
        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(linear)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""
        feat = self.features(x)
        return self.classifier(feat)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pre‑classifier feature vector."""
        return self.features(x)


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a configurable feed‑forward classifier and metadata.

    Returns
    -------
    network : nn.Module
        The instantiated model.
    encoding : Iterable[int]
        Feature indices used for data encoding (identity mapping).
    weight_sizes : Iterable[int]
        Number of trainable parameters per layer.
    observables : List[int]
        Dummy observable indices matching the quantum version.
    """
    network = QuantumClassifier(
        input_dim=num_features,
        depth=depth,
        hidden_dim=hidden_dim,
        dropout=dropout,
        batch_norm=batch_norm,
    )
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in network.parameters()]
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
