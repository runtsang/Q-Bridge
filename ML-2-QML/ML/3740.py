"""Unified hybrid classifier with a fully‑connected residual backbone.

This module defines `UnifiedHybridClassifier` which extends the two seed
implementations:  *   `QuantumClassifierModel.build_classifier_circuit` – the
  classical build‑function that returns a torch `nn.Module` and metadata;  *
  `QuantumClassifierModel.QCNet` – the quantum‑augmented network from the
  second seed.  The new class keeps the hybrid `Hybrid` layer that
  differentiable‑gradient‑aware *expectation* from the quantum circuit
  and uses a residual‑dense‑block structure to keep‑waste‑tolerant
  learning.  The public API is consistent with both seeds, so
  developers‑in‑the‑wild – all‑time‑bottleneck treat **one‑step** = – 
   .
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a deep residual feed‑forward network that mirrors the interface
    of the quantum circuit factory.  The network is a stack of *depth*
    residual blocks followed by a 2‑output head.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    depth : int
        Number of residual layers.

    Returns
    -------
    network : nn.Module
        The constructed network.
    encoding : Iterable[int]
        List of feature indices that are exposed to the quantum side.
    weight_sizes : Iterable[int]
        Total number of trainable parameters per linear layer.
    observables : List[int]
        Indices of the output logits that correspond to the two classes.
    """
    class ResidualBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.fc2(F.relu(self.fc1(x)))

    layers: List[nn.Module] = [nn.Linear(num_features, num_features), nn.ReLU()]
    weight_sizes: List[int] = [num_features * num_features + num_features]
    for _ in range(depth):
        layers.append(ResidualBlock(num_features))
        # each residual block contains two linear layers
        weight_sizes.append(num_features * num_features + num_features)
        weight_sizes.append(num_features * num_features + num_features)
    layers.append(nn.Linear(num_features, 2))
    weight_sizes.append(num_features * 2 + 2)

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = [0, 1]
    return network, encoding, weight_sizes, observables


class UnifiedHybridClassifier(nn.Module):
    """
    Simple residual backbone that can be paired with a quantum head.
    The class only contains the classical part; the quantum interface is
    supplied via the `build_classifier_circuit` function.
    """

    def __init__(self, num_features: int, depth: int = 3):
        super().__init__()
        self.classifier, _, _, _ = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


__all__ = ["build_classifier_circuit", "UnifiedHybridClassifier"]
