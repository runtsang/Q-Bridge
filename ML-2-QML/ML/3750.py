"""Hybrid quantum-inspired classifier with classical CNN backbone.

The model fuses a small convolutional feature extractor, a fully‑connected
projection, and a linear head mirroring the architecture of the quantum
counterpart.  A helper factory `build_classifier_circuit` returns a
classical feed‑forward network together with metadata that matches the
quantum build routine, enabling side‑by‑side benchmarking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class HybridQuantumClassifier(nn.Module):
    """
    Classical analogue of the quantum‑NAT encoder + variational layer.
    The network consists of a 2‑layer CNN, a projection to four features,
    batch‑norm, and a linear classifier head.
    """

    def __init__(self, num_features: int = 4, num_classes: int = 2, depth: int = 2) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # projection to the 4‑dimensional feature space
        self.proj = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_features),
        )
        self.norm = nn.BatchNorm1d(num_features)
        self.classifier_head = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.cnn(x)
        flattened = feats.view(bsz, -1)
        proj = self.proj(flattened)
        normed = self.norm(proj)
        logits = self.classifier_head(normed)
        return logits


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier mirroring the quantum helper.
    Returns a tuple of (network, encoding indices, weight‑sizes, observables).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding: List[int] = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]
