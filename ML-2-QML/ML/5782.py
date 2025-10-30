"""Enhanced classical classifier mirroring the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """
    Classical feed‑forward network that mimics the signature of the quantum
    `build_classifier_circuit` helper.

    Returns:
        network (nn.Module): The classifier.
        encoding (Iterable[int]): Feature indices used by the circuit.
        weight_sizes (Iterable[int]): Number of parameters per linear layer.
        observables (List[int]): Dummy observable indices for compatibility.
    """

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int,
                                 dropout: float = 0.0) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = [0, 1]  # placeholder for quantum observable indices

        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
