"""Classical classifier mirroring and extending the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel(nn.Module):
    """
    A residual neural network that emulates the interface of the quantum classifier.

    This class extends the original seed by adding:
    - Residual connections for better gradient flow.
    - Dropout for regularisation.
    - An explicit weight-size report for each layer.
    - A convenience `build_classifier_circuit` helper that returns the network,
      encoding indices, weight sizes, and output observables.
    """

    def __init__(self, num_features: int, depth: int, hidden_dim: int = 128,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        layers: List[nn.Module] = []
        in_dim = num_features

        for _ in range(depth):
            block = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, in_dim),
            )
            # Add residual connection
            layers.append(nn.Sequential(block, nn.Identity()))
            in_dim = hidden_dim

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)
        self.encoding = list(range(num_features))
        self.observables = [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        """
        for block in self.body:
            x = block[0](x) + x
        return self.head(x)

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int,
                                 hidden_dim: int = 128,
                                 dropout: float = 0.2) -> Tuple[
        nn.Module, Iterable[int], List[int], List[int]]:
        """
        Construct a residual network and return metadata mirroring the quantum API.
        """
        model = QuantumClassifierModel(num_features, depth,
                                       hidden_dim=hidden_dim,
                                       dropout=dropout)
        encoding = list(range(num_features))
        weight_sizes = [p.numel() for p in model.parameters()]
        observables = [0, 1]
        return model, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
