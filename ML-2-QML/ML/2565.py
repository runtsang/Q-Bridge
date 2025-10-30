"""Hybrid classifier combining a classical CNN backbone with a variational quantum circuit surrogate.

The module exposes a `QuantumHybridClassifier` class and a `build_classifier_circuit` function
that mirrors the quantum interface used in the QML counterpart.  The surrogate network
has the same number of trainable parameters as the quantum circuit, allowing direct
parameter sharing when training a hybrid model.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


class QuantumHybridClassifier(nn.Module):
    """Classical CNN + FC backbone followed by a variational classifier.

    The classifier part is a simple feed‑forward network with the same number of
    parameters as the quantum circuit produced by `build_classifier_circuit`.
    """

    def __init__(self, depth: int = 3) -> None:
        super().__init__()
        # Feature extractor (CNN)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Projection to 4‑dimensional feature vector
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

        # Classical surrogate of the quantum classifier
        layers = []
        in_dim = 4
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        proj = self.fc(flat)
        normed = self.norm(proj)
        out = self.classifier(normed)
        return out


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return a classical surrogate network and metadata matching the quantum circuit.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector (typically 4).
    depth : int
        Number of hidden layers in the surrogate network.
    """
    layers = []
    in_dim = num_features
    weight_sizes = []
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

    encoding = list(range(num_features))
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = ["QuantumHybridClassifier", "build_classifier_circuit"]
