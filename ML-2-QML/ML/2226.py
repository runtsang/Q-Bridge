"""Hybrid classical classifier combining CNN feature extraction and a depth‑controlled feed‑forward network.

The architecture is inspired by QuantumClassifierModel and QuantumNAT. It
provides a `build_classifier_circuit` helper that mirrors the original
interface but now returns a network built from the same CNN feature
extractor and a configurable classifier head.  The metadata (encoding,
weight sizes, observables) is retained for compatibility with downstream
experimentation pipelines.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


class HybridClassifier(nn.Module):
    """CNN + feed‑forward hybrid classifier."""

    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        # Feature extractor identical to the classical part of QuantumNAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flattened size: 16 * 7 * 7 = 784
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

        # Classifier head: depth‑controlled feed‑forward network
        self.classifier = self._build_classifier(num_features=4, depth=depth)

    @staticmethod
    def _build_classifier(num_features: int, depth: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        out = self.norm(out)
        logits = self.classifier(out)
        return logits

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """Return a feed‑forward classifier network and metadata."""
        network = HybridClassifier._build_classifier(num_features, depth)
        encoding = list(range(num_features))
        weight_sizes = [p.numel() for p in network.parameters()]
        observables = list(range(2))
        return network, encoding, weight_sizes, observables


__all__ = ["HybridClassifier"]
