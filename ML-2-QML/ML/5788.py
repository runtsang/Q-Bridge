"""Hybrid classical-quantum classifier combining feed‑forward network and feature extraction."""

from __future__ import annotations

from typing import Callable, Iterable, Tuple, List, Optional

import torch
import torch.nn as nn


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dims: Optional[List[int]] = None,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier mirroring the quantum interface.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input data.
    depth : int
        Number of hidden layers.
    hidden_dims : list[int] | None
        Optional explicit list of hidden layer sizes. If ``None`` a
        default of ``num_features`` is used for every hidden layer.

    Returns
    -------
    network : nn.Module
        Sequential network.
    encoding : list[int]
        Dummy encoding metadata (input indices).
    weight_sizes : list[int]
        Number of trainable parameters per layer.
    observables : list[int]
        Dummy observable metadata for compatibility.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    hidden_dims = hidden_dims or [num_features] * depth
    for h in hidden_dims:
        linear = nn.Linear(in_dim, h)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = h

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class HybridClassifier(nn.Module):
    """
    Hybrid classifier that optionally prepends a feature extractor
    (e.g. quantum kernel) before a classical feed‑forward network.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dims: Optional[List[int]] = None,
        feature_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor or (lambda x: x)
        self.network, _, _, _ = build_classifier_circuit(
            num_features, depth, hidden_dims
        )
        # If a feature extractor changes the dimensionality, we need to
        # adapt the input layer accordingly.
        # For simplicity we assume the extractor returns the same shape.
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, features)``.
        """
        features = self.feature_extractor(x)
        return self.network(features)


__all__ = ["build_classifier_circuit", "HybridClassifier"]
