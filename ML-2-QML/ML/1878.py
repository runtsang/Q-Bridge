"""Enhanced classical classifier with residual connections and optional attention."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

class QuantumClassifierModel:
    """
    Residual feed‑forward classifier that optionally applies a simple attention
    transformation.  The network mimics the interface of the quantum helper
    by exposing an encoding list, weight sizes and a set of observables.
    """

    def __init__(self, num_features: int, depth: int, use_attention: bool = False):
        self.num_features = num_features
        self.depth = depth
        self.use_attention = use_attention
        self.network, self.encoding, self.weight_sizes, self.observables = self.build_classifier_circuit()

    def build_classifier_circuit(
        self,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """
        Build a residual network with optional attention.

        Returns:
            network: nn.Sequential containing the layers.
            encoding: list of input indices (one per feature).
            weight_sizes: sizes of all weight tensors including biases.
            observables: dummy list mirroring the quantum observables.
        """
        layers: list[nn.Module] = []
        in_dim = self.num_features

        # Encoding indices: all input features are used
        encoding = list(range(self.num_features))

        weight_sizes: list[int] = []

        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = self.num_features

        if self.use_attention:
            # Simple attention-like linear transformation
            attn = nn.Linear(self.num_features, self.num_features)
            layers.append(attn)
            weight_sizes.append(attn.weight.numel() + attn.bias.numel())

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel"]
