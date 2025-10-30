"""Advanced classical classifier mirroring the quantum helper interface with optional regularization."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable, Optional

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """Factory for a feed‑forward classifier that mimics the quantum interface."""

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: nn.Module = nn.ReLU(),
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Construct a multi‑layer perceptron.

        Parameters
        ----------
        num_features: int
            Dimensionality of the input data.
        depth: int
            Number of hidden layers.
        dropout: float, optional
            Dropout probability applied after each activation.
        batch_norm: bool, optional
            Whether to insert BatchNorm1d after each linear layer.
        activation: nn.Module, optional
            Activation function to use.

        Returns
        -------
        network: nn.Sequential
            The constructed classifier.
        encoding: Iterable[int]
            Indices of input features (identity encoding).
        weight_sizes: Iterable[int]
            Total number of trainable parameters per layer.
        observables: List[int]
            Target classes (0 and 1).
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = [0, 1]
        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
