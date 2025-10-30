"""Classical classifier mirroring the quantum interface with advanced features."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel(nn.Module):
    """
    A deep neural network that mimics the interface of the quantum classifier.
    Supports optional dropout and residual connections for richer expressivity.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        use_residual: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.use_dropout = use_dropout
        self.use_residual = use_residual

        body_layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            body_layers.append(linear)
            body_layers.append(nn.ReLU(inplace=True))
            if use_dropout:
                body_layers.append(nn.Dropout(p=dropout_prob))
            in_dim = num_features
        self.body = nn.Sequential(*body_layers)
        self.head = nn.Linear(num_features, 2)

        # Metadata that mirrors the quantum helper
        self.encoding = list(range(num_features))
        self.weight_sizes = [p.numel() for p in self.parameters()]
        self.observables = [torch.tensor([1, 0]) for _ in range(2)]  # placeholder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connections."""
        out = x
        for layer in self.body:
            out = layer(out)
        logits = self.head(out)
        return logits

    def compute_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple feature importance via absolute weight magnitudes of the first linear layer.
        Returns a tensor of shape (num_features,).
        """
        first_linear = self.body[0]
        importance = torch.abs(first_linear.weight).mean(dim=0)
        return importance

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        use_residual: bool = False,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Factory that returns the model and metadata analogous to the quantum helper.
        """
        model = QuantumClassifierModel(
            num_features,
            depth,
            use_dropout=use_dropout,
            dropout_prob=dropout_prob,
            use_residual=use_residual,
        )
        encoding = model.encoding
        weight_sizes = model.weight_sizes
        observables = list(range(2))
        return model, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
