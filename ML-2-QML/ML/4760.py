"""ML implementation of QuantumClassifierModel.

This module defines a purely classical convolutional network that mirrors the
interface of its quantum counterpart.  It is useful for ablation studies,
baseline comparisons, and as a drop‑in replacement when quantum resources
are unavailable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Build a densely connected classifier that emulates the structure of the
    quantum ansatz.  The function returns the network, a list of feature
    indices that would be encoded in the quantum model, the weight counts
    per layer, and an observation list (dummy for the classical case).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU(inplace=True))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # placeholder for compatibility
    return network, encoding, weight_sizes, observables

class QuantumClassifierModel(nn.Module):
    """
    Classical convolutional classifier that shares the same public API as the
    quantum implementation.  It can be used as a baseline or as a component
    in hybrid training pipelines.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        # Determine the flattened feature size
        dummy_input = torch.zeros(1, 3, 32, 32)
        dummy_out = self.features(dummy_input)
        flat_dim = dummy_out.numel() // dummy_out.shape[0]
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
