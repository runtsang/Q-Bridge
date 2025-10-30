"""Hybrid classical regressor with metadata for quantum coupling.

This module defines a classical feed‑forward network that mirrors the quantum
``EstimatorQNN`` interface.  It exposes the network, the feature‑encoding
indices, the weight‑size profile, and a dummy observable list so that a
quantum wrapper can consume the same metadata without modification.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple, List

__all__ = ["EstimatorQNN", "build_classifier_circuit"]


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_sizes: Iterable[int] | None = None,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier and return metadata that
    mirrors the quantum implementation.

    Parameters
    ----------
    num_features: int
        Dimensionality of the input.
    depth: int
        Number of hidden layers.
    hidden_sizes: Iterable[int] | None
        Optional specification of hidden layer sizes.  If omitted, each hidden
        layer will have ``num_features`` neurons.

    Returns
    -------
    network: nn.Module
        Sequential classifier.
    encoding: Iterable[int]
        Indices of the input features that are directly mapped to the
        circuit parameters.
    weight_sizes: Iterable[int]
        Number of trainable parameters for each linear layer.
    observables: List[int]
        Dummy observable list; quantum code can replace this with Pauli
        operators while keeping the API identical.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    hidden_sizes = hidden_sizes or [num_features] * depth
    for size in hidden_sizes:
        linear = nn.Linear(in_dim, size)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = size

    head = nn.Linear(in_dim, 2)  # binary classification head
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))  # placeholder for quantum observables

    return network, encoding, weight_sizes, observables


class EstimatorQNN(nn.Module):
    """
    Classical estimator that mimics the quantum ``EstimatorQNN`` API.

    The network is a simple fully‑connected regressor with optional
    depth and hidden‑size customization.  It also computes a weight‑size
    profile and exposes a dummy observable list so that the same
    metadata can be passed to a quantum wrapper.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Tuple[int,...] | None = None,
        output_dim: int = 1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or (8, 4)
        layers: List[nn.Module] = []

        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Metadata
        self.weight_sizes = [
            m.weight.numel() + m.bias.numel()
            for m in self.net.modules()
            if isinstance(m, nn.Linear)
        ]
        self.encoding = list(range(input_dim))
        self.observables = list(range(output_dim))  # placeholder

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    def get_weight_sizes(self) -> List[int]:
        """Return the number of trainable parameters per linear layer."""
        return self.weight_sizes

    def get_encoding(self) -> List[int]:
        """Return the feature indices used for parameter encoding."""
        return self.encoding

    def get_observables(self) -> List[int]:
        """Return the observable placeholder list."""
        return self.observables
