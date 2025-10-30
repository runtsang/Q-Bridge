"""Hybrid‑classical classifier with configurable regularisation.

The class mirrors the original interface but adds dropout and optional batch‑norm
layers, making the network more robust to over‑fitting.  It also exposes helper
methods for extracting parameter counts and metadata, useful for downstream
experiments."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel:
    """Classical feed‑forward classifier with optional regularisation."""

    def __init__(
        self,
        num_features: int,
        depth: int,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        num_features:
            Dimensionality of the input feature vector.
        depth:
            Number of hidden layers.
        dropout:
            Drop‑out probability applied after every hidden layer.
        use_batchnorm:
            If True, insert a BatchNorm1d layer after each hidden layer.
        device:
            Target device for the model.
        """
        self.device = torch.device(device)
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        layers: List[nn.Module] = []
        in_dim = num_features

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)

        self.network = nn.Sequential(*layers).to(self.device)

        # Metadata
        self.encoding = list(range(num_features))
        self.weight_sizes = [
            layer.weight.numel() + layer.bias.numel()
            for layer in self.network.modules()
            if isinstance(layer, nn.Linear)
        ]
        self.observables = list(range(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x.to(self.device))

    def get_params(self) -> List[torch.Tensor]:
        """Return a list of trainable parameters."""
        return list(self.network.parameters())

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int, dropout: float = 0.0, use_batchnorm: bool = False
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Factory method matching the original signature but returning the extended
        network and metadata.
        """
        model = QuantumClassifierModel(
            num_features, depth, dropout=dropout, use_batchnorm=use_batchnorm
        )
        return model.network, model.encoding, model.weight_sizes, model.observables


__all__ = ["QuantumClassifierModel"]
