"""Unified classical classifier that mirrors the quantum interface.

This module defines UnifiedClassifier, a feed‑forward network with
configurable depth, activation, and dropout.  The static method
build_classifier_circuit returns the module together with metadata
that matches the quantum signature: encoding (feature indices),
weight_sizes (parameter counts) and observables (dummy indices).
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UnifiedClassifier"]

class UnifiedClassifier(nn.Module):
    """
    A depth‑scaled dense network with optional dropout and activation.

    Parameters
    ----------
    num_features: int
        Dimensionality of the input feature vector.
    depth: int
        Number of hidden layers.
    activation: str, optional
        Activation function; defaults to "relu".
    dropout: float, optional
        Dropout probability; 0.0 disables dropout.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_features
        act_fn = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "sigmoid": nn.Sigmoid,
        }.get(activation.lower(), nn.Identity)
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Metadata that mimics the quantum interface
        self.encoding = list(range(num_features))
        self.weight_sizes = [p.numel() for p in self.parameters()]
        self.observables = list(range(2))  # dummy observables for compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        *,
        activation: str | None = None,
        dropout: float | None = None,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """
        Build a feed‑forward network and return it together with
        encoding, weight_sizes and dummy observables.

        The returned signature matches the quantum build_classifier_circuit
        for seamless replacement.
        """
        act = activation if activation is not None else "relu"
        drop = dropout if dropout is not None else 0.0
        model = UnifiedClassifier(num_features, depth, act, drop)
        return model, model.encoding, model.weight_sizes, model.observables
