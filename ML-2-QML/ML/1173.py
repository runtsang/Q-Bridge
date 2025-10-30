"""Classical classifier factory with advanced architecture mirroring the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn

class ClassifierFactory:
    """Factory for building a classical feed‑forward classifier.

    The interface matches the quantum variant: ``build_classifier_circuit`` returns
    a tuple of (model, encoding, weight_sizes, observables).  The returned
    observable list is a placeholder for the two class logits.
    """

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Construct a deep residual MLP with batch‑norm and dropout.

        Args:
            num_features: Number of input features.
            depth: Number of hidden layers.

        Returns:
            model: nn.Sequential instance.
            encoding: List of feature indices used for the input.
            weight_sizes: List of parameter counts per linear layer.
            observables: List of dummy observable identifiers (class indices).
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.BatchNorm1d(num_features), nn.ReLU(), nn.Dropout(p=0.2)])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        model = nn.Sequential(*layers)
        observables = [0, 1]  # placeholder class indices
        return model, encoding, weight_sizes, observables

__all__ = ["ClassifierFactory"]
