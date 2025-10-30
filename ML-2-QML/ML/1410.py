"""Enhanced classical classifier mirroring the quantum helper interface with residual blocks and dropout."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """
    Classical feed‑forward network with residual connections, batch‑norm and dropout.
    Mirrors the signature of the quantum helper so that it can be swapped at runtime.
    """

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
        """
        Construct a residual network with `depth` blocks. Each block contains
        Linear → BatchNorm → ReLU → Linear → BatchNorm → ReLU → Dropout.
        The output head is a Linear layer mapping to two classes.

        Returns:
            network: nn.Sequential model
            encoding: list of input feature indices (identity mapping)
            weight_sizes: number of trainable parameters per linear layer
            observables: class indices (0 and 1)
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            # First linear layer
            linear1 = nn.Linear(in_dim, num_features)
            bn1 = nn.BatchNorm1d(num_features)
            # Second linear layer
            linear2 = nn.Linear(num_features, num_features)
            bn2 = nn.BatchNorm1d(num_features)
            dropout = nn.Dropout(p=0.2)

            # Residual block
            block = nn.Sequential(
                linear1,
                bn1,
                nn.ReLU(inplace=True),
                linear2,
                bn2,
                nn.ReLU(inplace=True),
                dropout
            )
            layers.append(block)

            # Residual connection: identity
            layers.append(nn.Identity())

            weight_sizes.append(linear1.weight.numel() + linear1.bias.numel())
            weight_sizes.append(linear2.weight.numel() + linear2.bias.numel())

            in_dim = num_features

        # Head
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = [0, 1]
        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
